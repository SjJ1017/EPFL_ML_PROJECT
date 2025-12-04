from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.PdfToken import PdfToken
from pdf_features.PdfPage import PdfPage
from tqdm import tqdm
import numpy as np
from pathlib import Path
from dataloader import DocLayNetDataset
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from time import time
from model import *

class FeatureExtractorSegment(FeatureExtractor):
        
    def extract_statistical_features(self, token, page, prev_token=None, next_token=None):

        features = []
        
        content = token.content
        features.extend([
            token.token_type.get_index(),
            len(content),  
            len(content.split()),  
            sum(1 for c in content if c.isupper()) / (len(content) + 1e-6), 
            sum(1 for c in content if c.isdigit()) / (len(content) + 1e-6), 
            sum(1 for c in content if c.isspace()) / (len(content) + 1e-6), 
            1.0 if content.isupper() else 0.0, 
            1.0 if content[0].isupper() else 0.0,  
            1.0 if any(c.isdigit() for c in content) else 0.0, 
            1.0 if any(c in '.,;:!?' for c in content) else 0.0, 
        ])
        
        font_id = float(token.font.font_id)
        font_size = float(token.font.font_size)
        features.extend([
            font_id,
            font_size,
            np.log(font_size + 1),  # log scale
        ])
        
        box = token.bounding_box
        page_width = page.page_width
        page_height = page.page_height
        
        left_norm = box.left / page_width
        top_norm = box.top / page_height
        right_norm = box.right / page_width
        bottom_norm = box.bottom / page_height
        
        width_norm = (box.right - box.left) / page_width
        height_norm = (box.top - box.bottom) / page_height
        
        features.extend([
            left_norm, top_norm, right_norm, bottom_norm,
            width_norm, height_norm,
            left_norm + width_norm / 2,  
            top_norm - height_norm / 2,  
            width_norm * height_norm, 
            width_norm / (height_norm + 1e-6), 
        ])
        
        features.extend([
            left_norm,  
            1 - right_norm, 
            top_norm,  
            1 - bottom_norm, 
            1.0 if left_norm < 0.1 else 0.0, 
            1.0 if right_norm > 0.9 else 0.0, 
            1.0 if top_norm < 0.1 else 0.0,  
            1.0 if bottom_norm > 0.9 else 0.0,  
            1.0 if 0.4 < left_norm + width_norm / 2 < 0.6 else 0.0, 
        ])
        
        if prev_token:
            prev_box = prev_token.bounding_box

            horizontal_dist = (box.left - prev_box.right) / page_width
            vertical_dist = (prev_box.top - box.top) / page_height
            same_line = 1.0 if abs(vertical_dist) < 0.01 else 0.0
            font_size_diff = font_size - float(prev_token.font.font_size)
            
            features.extend([
                prev_token.token_type.get_index(),
                horizontal_dist,
                vertical_dist,
                same_line,
                font_size_diff,
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        if next_token:
            next_box = next_token.bounding_box
            horizontal_dist = (next_box.left - box.right) / page_width
            vertical_dist = (box.top - next_box.top) / page_height
            same_line = 1.0 if abs(vertical_dist) < 0.01 else 0.0
            font_size_diff = float(next_token.font.font_size) - font_size
            
            features.extend([
                next_token.token_type.get_index(),
                horizontal_dist,
                vertical_dist,
                same_line,
                font_size_diff,
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return features
    
    def get_page_features(self):
        pages_features = []
        page_targets = []
        page_id = 0
        for page in self.loop_token_features():
            page_id += 1
            print("Processing page:", page_id)
            if page_id > 10:
                return pages_features, page_targets
            tokens = page.tokens
            if len(tokens) == 0:
                continue
                
            targets = []
            feature_rows = []
            
            for i, token in enumerate(tokens):
                prev_token = tokens[i-1] if i > 0 else None
                next_token = tokens[i+1] if i < len(tokens) - 1 else None
                
                features = self.extract_statistical_features(
                    token, page, prev_token, next_token
                )
                
                feature_rows.append(features)
                targets.append(token.prediction)
            
            pages_features.append(feature_rows)
            page_targets.append(targets)
        
        return pages_features, page_targets
class TransformerTaggerSegment(TransformerTagger, FeatureExtractorSegment):
    def __init__(self, *args, **kwargs):
        TransformerTagger.__init__(self, *args, **kwargs)
        # find pdfs_features in args or kwargs
        pdfs_features = kwargs.get("pdfs_features", None)
        FeatureExtractorSegment.__init__(self, pdfs_features)

    def labeled_features(self):
        if self.pdfs_features is None:
            raise ValueError("Feature extractor not loaded with PdfFeatures but with DocLayNetDataset. This method is only for PdfFeatures.")
        labels = self.predict()
        assert len(labels) == sum(len(page.tokens) for pdf in self.pdfs_features for page in pdf.pages)
        for pdf in self.pdfs_features:
            for page in pdf.pages:
                for token in page.tokens:
                    pred = labels.pop(0)
                    token.prediction = pred
        return self.pdfs_features
    
if __name__ == "__main__":
    import pickle
    import os
    import argparse

    parser = argparse.ArgumentParser("DocLayNet Training")
    parser.add_argument("--data_path", type=str, help="Path to the DocLayNet dataset")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"], help="Device to use for training")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument('--split', type=str, default="test", help="Dataset split to use (train/val/test)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--features_cache", type=str, default=".cache/features_cache_segmented.pkl", help="Path to cache extracted features")
    parser.add_argument("--data_cache", type=str, default=".cache/data_cache_segmented.pkl", help="Path to cache dataset")
    parser.add_argument("--train_split", type=float, default=0.8, help="Proportion of data to use for training")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension size of the Transformer")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of Transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads in the Transformer")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for optimizer")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focusing parameter for Focal Loss")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--model_save_path", type=str, default="best_model.pth", help="Path to save the best model")
    args = parser.parse_args()


    torch.manual_seed(42)
    np.random.seed(42)
    
    num_classes = 2
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    features_cache_file = args.features_cache

    if os.path.exists(features_cache_file):
        print(f"Found feature cache: {features_cache_file}, Loading cached features...")
        with open(features_cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            pages_features = cache_data['pages_features']
            pages_targets = cache_data['pages_targets']
            feature_dim = cache_data['feature_dim']
        print(f"Loaded.")
    else:
        print("Loading dataset...")
        dataset = DocLayNetDataset(split=args.split, rewrite_storage=False)
        if os.path.exists(args.data_cache):
            print("Loading data cache...")
            dataset.load_cache(input_path=args.data_cache)
        
        feature_extractor = FeatureExtractorSegment(dataset)
        if not os.path.exists(args.data_cache):
            dataset.save_cache(output_path=args.data_cache)
        pages_features, pages_targets = feature_extractor.get_page_features()
        feature_dim = len(pages_features[0][0])
        
        cache_data = {
            'pages_features': pages_features,
            'pages_targets': pages_targets,
            'feature_dim': feature_dim
        }
        with open(features_cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    
    import random
    features_and_targets = list(zip(pages_features, pages_targets))
    random.shuffle(features_and_targets)
    total = len(features_and_targets)
    training_data = features_and_targets[:int(args.train_split * total)]
    validation_data = features_and_targets[int(args.train_split * total):]
    print(f"\nTraining samples: {len(training_data)}, Validation samples: {len(validation_data)}")
    
    class_weights = compute_class_weights(pages_targets, num_classes, device)
    model = TransformerTagger(
        input_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        output_dim=num_classes,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    criterion = CombinedLoss(
        num_classes=num_classes,
        alpha=class_weights,
        gamma=args.gamma,
        label_smoothing=0.1,
        ignore_index=-1
    )
    
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    best_accuracy = 0.0
    losses_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        
        pbar = tqdm(range(0, len(training_data), BATCH_SIZE), desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i in pbar:
            batch_data = training_data[i:i+BATCH_SIZE]
            
            x, y, masks, lengths = collate_batch(batch_data, feature_dim)
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
        
        print(f"\nEvaluating epoch {epoch+1}...")
        accuracy, class_acc, _, _ = evaluate_model(model, validation_data, feature_dim, device, num_classes)
        
        print(f"Epoch {epoch+1} - Overall Accuracy: {accuracy:.4f}")
        print("Per-class accuracy:")
        for i, acc in enumerate(class_acc):
            print(f"  Class {i}: {acc:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), args.model_save_path)
    
    print(f"\nTraining completed. Best accuracy: {best_accuracy:.4f}")
