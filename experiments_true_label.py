import itertools
import json
import os
import subprocess
import re
from datetime import datetime
import pandas as pd
from model import *
from model_segment import *
from dataloader import *
import numpy as np
import torch
import pickle
import random
from utils import *



def train(split = 'test', num_classes = 11, train_type = 'token', random_seed=42, device ='cuda', cache_dir='.cache', data_cache='data_cache.pkl', train_split=0.8, hidden_dim=512, num_layers=6, num_heads = 8, max_len=512, dropout = 0.2, weight_decay =0.01, learning_rate=2e-4, gamma=2.0, batch_size=16, epochs=30, model_save_dir='models', end_idx = 10):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    os.makedirs(f"{cache_dir}/{train_type}", exist_ok=True)
    feature_path = f"{cache_dir}/{train_type}/features.pkl"
    cache_path = f"{cache_dir}/{train_type}/cache.pkl"
 
    print("Loading dataset...")
    dataset = DocLayNetDataset(split=split, rewrite_storage=False, end_idx=end_idx)
    assert train_type in ['token', 'gold_seg', 'noisy_seg'], "Invalid type specified."
    feature_extractor = FeatureExtractor(dataset) if train_type == 'token' else FeatureExtractorSegment(dataset)
    if os.path.exists(feature_path):
        print(f"Found feature cache: {feature_path}, Loading cached features...")
        with open(feature_path, 'rb') as f:
            cache_data = pickle.load(f)
            pages_features = cache_data['pages_features']
            pages_targets = cache_data['pages_targets']
            feature_dim = cache_data['feature_dim']
        print(f"Loaded.")
    else:

        if os.path.exists(cache_path):
            print("Loading data cache...")
            dataset.load_cache(input_path=cache_path)

        feature_extractor = FeatureExtractor(dataset) if train_type == 'token' else FeatureExtractorSegment(dataset)
        if not os.path.exists(cache_path):
            dataset.save_cache(output_path=cache_path)
        pages_features, pages_targets = feature_extractor.get_page_features()
        feature_dim = len(pages_features[0][0])
        cache_data = {
        'pages_features': pages_features,
        'pages_targets': pages_targets,
        'feature_dim': feature_dim
    }
    with open(feature_path, 'wb') as f:
        pickle.dump(cache_data, f)

    pages_features = [p[:max_len] for p in pages_features]
    if train_type == 'noisy_seg':
        print("Replacing with predictions")
        token_type_model = TransformerTagger(
            input_dim=feature_dim - 3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            output_dim=len(TokenType),
            dropout=dropout,
            max_seq_len=max_len,
        ).to(device)
        token_type_model.load_state_dict(torch.load(f"{model_save_dir}/H{hidden_dim}_L{num_layers}_G{str(gamma).replace('.', '')}/model_token_epoch{epochs}.pth", map_location=device))
        # token_type_model.load_state_dict(torch.load("models/best_larger_train_deep.pth", map_location=device)) # for testing
        token_type_model.eval()
        with open(f"{cache_dir}/token/features.pkl", 'rb') as f:
            features = pickle.load(f)
        features, _feature_dim = features.get('pages_features'), features.get('feature_dim')
        indices, length = feature_extractor.identify_adding_positions()
        print(f"Identified {len(indices)} positions to add replace")
        assert length == _feature_dim + 3, "Length mismatch when adding noisy segment features."
        pages_labels = []
        ls = []
        for feature in tqdm(features):
            feature = torch.tensor(feature[:max_len], dtype=torch.float32).to(device)
            prediction = token_type_model(feature)
            ls.append(len(feature))
            labels = torch.argmax(prediction, dim=-1)
            pages_labels.extend(labels.cpu().numpy().tolist())
        predictions = pages_labels
        prev_predictions = [0.0] + predictions[:-1]
        next_predictions = predictions[1:] + [0.0]
        each_length = [min(max_len, len(p)) for p in pages_features]
        # print(each_length)
        # print(ls)
        assert sum(each_length) == len(predictions), f"Length mismatch in predictions: {sum(each_length)} != {len(predictions)}"
        pages_features_stacked = np.vstack(pages_features)
        pages_features_stacked = replace_columns(pages_features_stacked, [predictions, prev_predictions, next_predictions], ks=indices)

        pages_features = []
        _start = 0
        for k in each_length:
            pages_features.append(pages_features_stacked[_start:_start+k, :])
            _start += k

    features_and_targets = list(zip(pages_features, pages_targets))
    random.shuffle(features_and_targets)
    total = len(features_and_targets)
    training_data = features_and_targets[:int(train_split * total)]
    validation_data = features_and_targets[int(train_split * total):]
    print(f"\nTraining samples: {len(training_data)}, Validation samples: {len(validation_data)}")
    
    class_weights = compute_class_weights(pages_targets, num_classes, device)
    model = TransformerTagger(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        # num_layers=4, for testing ---
        num_heads=num_heads,
        output_dim=num_classes,
        dropout=dropout,
        max_seq_len=max_len, 
        # max_seq_len=2000, for testing ---
    ).to(device)
    # model.load_state_dict(torch.load("models/best_model_segment_.pth", map_location=device)) for testing ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    criterion = CombinedLoss(
        num_classes=num_classes,
        alpha=class_weights,
        gamma=gamma,
        label_smoothing=0.1,
        ignore_index=-1
    )
    
    BATCH_SIZE = batch_size
    EPOCHS = epochs
    
    best_accuracy = 0.0
    losses_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        
        pbar = tqdm(range(0, len(training_data), BATCH_SIZE), desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i in pbar:
            batch_data = training_data[i:i+BATCH_SIZE]
            
            x, y, masks, lengths = collate_batch(batch_data, feature_dim, max_seq_len=max_len)
            x, y, masks = x.to(device), y.to(device), masks.to(device)
            
            optimizer.zero_grad()
            pred = model(x, mask=masks)
            
            pred_flat = pred.view(-1, num_classes)
            y_flat = y.view(-1)
            
            loss = criterion(pred_flat, y_flat)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
            losses_history.append(loss.item())
            
            if len(epoch_losses) > 0:
                pbar.set_postfix({'loss': f'{np.mean(epoch_losses[-100:]):.4f}'})
        
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            if train_type != 'noisy_seg':
                # mkdir "H{hidden_dim}_L{num_layers}_G{gamma}"
                os.makedirs(f"{model_save_dir}/H{hidden_dim}_L{num_layers}_G{str(gamma).replace('.', '')}", exist_ok=True)
                torch.save(model.state_dict(), f"{model_save_dir}/H{hidden_dim}_L{num_layers}_G{str(gamma).replace('.', '')}/model_{train_type}_epoch{epoch+1}.pth")
            accuracy, class_acc, _, _ = evaluate_model(model, validation_data, feature_dim, device, num_classes, max_len=max_len)
            print(f"Epoch {epoch+1} Validation Accuracy: {accuracy:.4f}")
    if train_type == 'noisy_seg': # only save at the end for noisy_seg
        os.makedirs(f"{model_save_dir}/H{hidden_dim}_L{num_layers}_G{str(gamma).replace('.', '')}", exist_ok=True)
        torch.save(model.state_dict(), f"{model_save_dir}/H{hidden_dim}_L{num_layers}_G{str(gamma).replace('.', '')}/model_{train_type}_epoch{epochs}.pth")
        accuracy, class_acc, _, _ = evaluate_model(model, validation_data, feature_dim, device, num_classes, max_len=max_len)
        print(f"Epoch {epochs} Final Validation Accuracy: {accuracy:.4f}")

# test_results
def aggregating_tokens(tokens, segs):
    assert len(tokens) == len(segs), "Length mismatch between tokens and segments."

    curr_tokens = []
    starting_index = 0
    for i, token, seg in zip(range(len(tokens)), tokens, segs):
        if seg == 0:
            curr_tokens.append(token)
        else:
            curr_tokens.append(token)
            majority_token = max(set(curr_tokens), key=curr_tokens.count)
            tokens[starting_index:i+1] = [majority_token] * (i - starting_index + 1)
            starting_index = i + 1
            curr_tokens = []

    if curr_tokens:
        majority_token = max(set(curr_tokens), key=curr_tokens.count)
        tokens[starting_index:len(tokens)] = [majority_token] * (len(tokens) - starting_index)
    return tokens, segs

def _evaluate_model(model, segment_model, validation_data, device, num_classes, max_len=512, extractor=None):
    model.eval()
    segment_model.eval()
    all_preds_token = []
    all_preds_seg = []
    all_targets_token = []
    all_targets_seg = []
    
    with torch.no_grad():
        for page_features, targets_token, targets_seg in validation_data:
            x = torch.tensor(page_features, dtype=torch.float32).to(device)
            y_token = torch.tensor(targets_token, dtype=torch.long).to(device)
            # print(x.shape)
            if x.shape[0] > max_len:
                x = x[:max_len, :] 

            x = x.unsqueeze(0)
            
            if y_token.shape[0] > max_len:
                y_token = y_token[:max_len]


            token_pred = model(x)
            _, predicted = torch.max(token_pred.squeeze(0), dim=1)

            all_preds_token.extend(predicted.cpu().tolist())
            all_targets_token.extend(y_token.tolist())

    correct = sum(p == t for p, t in zip(all_preds_token, all_targets_token))
    accuracy = correct / len(all_targets_token)

    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    for p, t in zip(all_preds_token, all_targets_token):
        class_total[t] += 1
        if p == t:
            class_correct[t] += 1
    
    class_acc = [class_correct[i] / (class_total[i] + 1e-6) for i in range(num_classes)]
    indices, length = extractor.identify_adding_positions()
    print(f"Identified {len(indices)} positions to add replace")
    all_preds_token_prev = [0.0] + all_preds_token[:-1]
    all_preds_token_next = all_preds_token[1:] + [0.0]
    each_len = [min(len(p), max_len) for p, _,_ in validation_data]
    assert sum(each_len) == len(all_preds_token), "Length mismatch in predictions."
    pages_features_stacked = np.vstack([p[:max_len] for p, _,_ in validation_data])
    pages_features_stacked = insert_columns(pages_features_stacked, [all_preds_token, all_preds_token_prev, all_preds_token_next], ks=indices)
    # print(pages_features_stacked.shape)
    pages_features = []
    _start = 0
    for k in each_len:
        pages_features.append(pages_features_stacked[_start:_start+k, :])
        _start += k
    
    for i, features in enumerate(pages_features):
        targets_seg = validation_data[i][2]
        x_seg = torch.tensor(features, dtype=torch.float32).to(device)
        y_seg = torch.tensor(targets_seg, dtype=torch.long).to(device)
        if x_seg.shape[0] > max_len:
            x_seg = x_seg[:max_len, :]
        if y_seg.shape[0] > max_len:
            y_seg = y_seg[:max_len]
        x_seg = x_seg.unsqueeze(0)
        seg_pred = segment_model(x_seg)
        _, predicted_seg = torch.max(seg_pred.squeeze(0), dim=1)
        all_preds_seg.extend(predicted_seg.cpu().tolist())
        all_targets_seg.extend(y_seg.tolist())
    
    seg_correct = sum(p == t for p, t in zip(all_preds_seg, all_targets_seg))
    seg_accuracy = seg_correct / len(all_targets_seg)

    seg_class_correct = [0] * 2
    seg_class_total = [0] * 2
    for p, t in zip(all_preds_seg, all_targets_seg):
        seg_class_total[t] += 1
        if p == t:
            seg_class_correct[t] += 1
    seg_class_accuracy = [seg_class_correct[i] / (seg_class_total[i] + 1e-6) for i in range(2)]

    # print(all_preds_token)

    all_preds_token, all_preds_seg = aggregating_tokens(all_preds_token, all_preds_seg)
    correct_token = sum(p == t for p, t in zip(all_preds_token, all_targets_token))
    accuracy_token = correct_token / len(all_targets_token)

    class_correct_token = [0] * num_classes
    class_total_token = [0] * num_classes
    
    for p, t in zip(all_preds_token, all_targets_token):
        class_total_token[t] += 1
        if p == t:
            class_correct_token[t] += 1
    
    class_acc_token = [class_correct_token[i] / (class_total_token[i] + 1e-6) for i in range(num_classes)]
    # print(all_preds_token)
    # print(all_preds_seg)

    return accuracy, class_acc, all_preds_token, all_targets_token, seg_accuracy, seg_class_accuracy, accuracy_token, class_acc_token



def evaluate(split = 'validation', num_classes = 11, val_type = 'token', random_seed=42, device ='cuda', validation_cache = '.cache/val.pkl', train_split=0.8, hidden_dim=512, num_layers=6, num_heads = 8, max_len=512, dropout = 0.2, weight_decay =0.01, learning_rate=2e-4, gamma=2.0, batch_size=16, epochs=30, model_save_dir='models', end_idx = 10, results_dir = '.results'):
    extractor = FeatureExtractor(DocLayNetDataset(split=split, rewrite_storage=False, end_idx=end_idx))
    extractor_segment = FeatureExtractorSegment(DocLayNetDataset(split=split, rewrite_storage=False, end_idx=end_idx))

    if not os.path.exists(validation_cache):
        pages_features, pages_targets_token = extractor.get_page_features()
        pages_features_seg, pages_targets_seg = extractor_segment.get_page_features()
        saving = {
            'pages_features': pages_features,
            'pages_targets_token': pages_targets_token,
            'pages_features_seg': pages_features_seg,
            'pages_targets_seg': pages_targets_seg
        }
        pickle.dump(saving, open(validation_cache, 'wb'))
    else:
        saving = pickle.load(open(validation_cache, 'rb'))
        pages_features = saving['pages_features']
        pages_targets_token = saving['pages_targets_token']
        pages_features_seg = saving['pages_features_seg']
        pages_targets_seg = saving['pages_targets_seg']
        
    validation_data = list(zip(pages_features, pages_targets_token, pages_targets_seg))
    model = TransformerTagger(
        input_dim=len(pages_features[0][0]),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        output_dim=num_classes,
        dropout=dropout,
        max_seq_len=max_len, 
    ).to(device)

    model.load_state_dict(torch.load(f"{model_save_dir}/H{hidden_dim}_L{num_layers}_G{str(gamma).replace('.', '')}/model_token_epoch{epochs}.pth", map_location=device))

    model_segment = TransformerTagger(
        input_dim=len(pages_features_seg[0][0]),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        output_dim=2,
        dropout=dropout,
        max_seq_len=max_len, 
    ).to(device)

    model_segment.load_state_dict(torch.load(f"{model_save_dir}/H{hidden_dim}_L{num_layers}_G{str(gamma).replace('.', '')}/model_{val_type}_epoch{epochs}.pth", map_location=device))

    accuracy, class_acc, all_preds_token, all_targets_token, seg_accuracy, seg_class_accuracy, accuracy_token, class_acc_token = _evaluate_model(model, model_segment, validation_data, device, num_classes, max_len=max_len, extractor=extractor_segment)
    
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        with open(f"{results_dir}/results_{val_type}_L{num_layers}_H{hidden_dim}_G{str(gamma).replace('.', '')}_epoch{epochs}.txt", "w") as f:
            f.write(f"Validation Accuracy: {accuracy:.4f}\n")
            f.write(f"Validation Class Accuracy: {class_acc}\n")
            f.write(f"Segmentation Accuracy: {seg_accuracy:.4f}\n")
            f.write(f"Segmentation Class Correct: {seg_class_accuracy}\n")
            f.write(f"Token Accuracy: {accuracy_token:.4f}\n")
            f.write(f"Token Class Accuracy: {class_acc_token}\n")


    return accuracy, class_acc, all_preds_token, all_targets_token, seg_accuracy, seg_class_accuracy, accuracy_token, class_acc_token

import argparse

parser = argparse.ArgumentParser(description="Train and evaluate the model")
parser.add_argument("--num_layers", type=int, default=6, help="Number of layers in the model")
parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension of the model")
parser.add_argument("--gamma", type=float, default=2.0, help="Gamma value for focal loss")
parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
parser.add_argument("--train_samples", type=int, default=None, help="Number of training samples (None for full set)")
parser.add_argument("--val_num", type=int, default=2000, help="Number of validation samples")
args = parser.parse_args()

# PARAMETERS

NUM_LAYERS = args.num_layers
HIDDEN_DIM = args.hidden_dim
GAMMA = args.gamma
EPOCHS = args.epochs
TRAIN_SAMPLES = args.train_samples
VAL_NUM = args.val_num

# First, train the token_type model
skip_train = False
if NUM_LAYERS == 6 and HIDDEN_DIM == 512 and GAMMA == 2.0 and EPOCHS ==30:
    skip_train = True
    print("Skipping token_type model training as this configuration is already trained.")

if not skip_train:
    train(
        split="test",
        num_classes=11,
        random_seed=42,
        device="cuda",
        train_type="token",
        cache_dir=".cache",
        train_split=0.8,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        max_len=512,
        learning_rate=2e-4,
        gamma=GAMMA,
        batch_size=16,
        epochs=EPOCHS,
        model_save_dir="models",
        end_idx=TRAIN_SAMPLES,
    )

    # Then, train the segmenting model
    train(
        split="test",
        num_classes=2,
        random_seed=42,
        device="cuda",
        train_type="gold_seg", # or "seg_noisy"
        cache_dir=".cache",
        train_split=0.8,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        max_len=512,
        learning_rate=2e-4,
        gamma=GAMMA,
        batch_size=16,
        epochs=EPOCHS,
        model_save_dir="models",
        end_idx=TRAIN_SAMPLES,
    )

accuracy, class_acc, all_preds_token, all_targets_token, seg_accuracy, seg_class_accuracy, accuracy_token, class_acc_token = evaluate(
    split="validation",
    num_classes=11,
    random_seed=42,
    device="cuda",
    val_type="gold_seg", # or "seg_noisy"
    validation_cache = '.cache/val.pkl',
    train_split=0.8,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    max_len=512,
    learning_rate=2e-4,
    gamma=GAMMA,
    batch_size=16,
    epochs=EPOCHS,
    model_save_dir="models",
    end_idx=VAL_NUM,
    results_dir ='.results'
)
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Class Accuracy: {class_acc}")
print(f"Segmentation Accuracy: {seg_accuracy:.4f}")
print(f"Segmentation Class Correct: {seg_class_accuracy}")
print(f"Token Accuracy: {accuracy_token:.4f}")
print(f"Token Class Accuracy: {class_acc_token}")