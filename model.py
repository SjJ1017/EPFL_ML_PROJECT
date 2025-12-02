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

class ImprovedFeatureExtractor:

    def __init__(self, dataset: DocLayNetDataset):
        self.dataset = dataset

        self.font_stats = None
        
    def extract_statistical_features(self, token, page, prev_token=None, next_token=None):

        features = []
        
        content = token.content
        features.extend([
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
                horizontal_dist,
                vertical_dist,
                same_line,
                font_size_diff,
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        if next_token:
            next_box = next_token.bounding_box
            horizontal_dist = (next_box.left - box.right) / page_width
            vertical_dist = (box.top - next_box.top) / page_height
            same_line = 1.0 if abs(vertical_dist) < 0.01 else 0.0
            font_size_diff = float(next_token.font.font_size) - font_size
            
            features.extend([
                horizontal_dist,
                vertical_dist,
                same_line,
                font_size_diff,
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        return features
    
    def get_page_features(self):
        pages_features = []
        page_targets = []
        
        for page in self.loop_token_features():
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
                targets.append(token.token_type.get_index())
            
            pages_features.append(feature_rows)
            page_targets.append(targets)
        
        return pages_features, page_targets
    
    def loop_token_features(self):
        for pdf_features in tqdm(self.dataset, desc="Extracting features"):
            for page in pdf_features.pages:
                if not page.tokens:
                    continue
                yield page


class ImprovedTransformerTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim=256, num_layers=4, num_heads=8, 
                 output_dim=11, dropout=0.2, max_seq_len=2000):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # LayerNorm and Dropout
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.pos_embedding = self._create_positional_encoding(max_seq_len, hidden_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True  
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self._init_weights()
    
    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe, requires_grad=True)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, _ = x.shape
        
        x_emb = self.input_proj(x)  # (batch_size, seq_len, hidden_dim)
        
        x_emb = x_emb + self.pos_embedding[:seq_len].unsqueeze(0)
        
        if mask is not None:
            attn_mask = mask.float().masked_fill(mask, float('-inf')).masked_fill(~mask, 0.0)
        else:
            attn_mask = None
        
        encoded = self.encoder(x_emb, src_key_padding_mask=mask)
        
        out = self.output_proj(encoded)
        
        if squeeze_output:
            out = out.squeeze(0)
        
        return out


class CombinedLoss(nn.Module):
    
    def __init__(self, num_classes, alpha=None, gamma=2.0, label_smoothing=0.1, ignore_index=-1):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        valid_mask = (targets != self.ignore_index)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        logits_valid = logits[valid_mask]
        targets_valid = targets[valid_mask]
        
        # Label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(logits_valid)
            true_dist.fill_(self.label_smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets_valid.unsqueeze(1), 1.0 - self.label_smoothing)
        
        log_probs = torch.log_softmax(logits_valid, dim=-1)
        
        # CE loss with label smoothing
        ce_loss = -(true_dist * log_probs).sum(dim=-1)
        
        # Focal loss weight
        probs = torch.softmax(logits_valid, dim=-1)
        pt = probs.gather(1, targets_valid.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        
        # Class weight (alpha)
        if self.alpha is not None:
            alpha_t = self.alpha[targets_valid]
            loss = alpha_t * focal_weight * ce_loss
        else:
            loss = focal_weight * ce_loss
        
        return loss.mean()


def collate_batch(batch_data, feature_dim):
    batch_x = []
    batch_y = []
    lengths = []
    
    for page_features, targets in batch_data:
        x = torch.tensor(page_features, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.long)
        batch_x.append(x)
        batch_y.append(y)
        lengths.append(len(targets))
    
    max_len = max(lengths)
    padded_x = []
    padded_y = []
    masks = []
    
    for x, y, length in zip(batch_x, batch_y, lengths):
        pad_size = max_len - length
        if pad_size > 0:
            x_padded = torch.cat([x, torch.zeros(pad_size, feature_dim)], dim=0)
            y_padded = torch.cat([y, torch.full((pad_size,), -1, dtype=torch.long)])
            mask = torch.cat([torch.zeros(length, dtype=torch.bool), 
                            torch.ones(pad_size, dtype=torch.bool)])
        else:
            x_padded = x
            y_padded = y
            mask = torch.zeros(length, dtype=torch.bool)
        
        padded_x.append(x_padded)
        padded_y.append(y_padded)
        masks.append(mask)
    
    batch_x = torch.stack(padded_x)  # (batch_size, max_len, feature_dim)
    batch_y = torch.stack(padded_y)  # (batch_size, max_len)
    masks = torch.stack(masks)  # (batch_size, max_len)
    
    return batch_x, batch_y, masks, lengths


def compute_class_weights(pages_targets, num_classes, device):
    label_counts = torch.zeros(num_classes, dtype=torch.float32)
    total = 0
    for labels in pages_targets:
        for lbl in labels:
            label_counts[lbl] += 1
            total += 1
    
    class_weights = total / (num_classes * (label_counts + 1e-6))
    class_weights = class_weights.to(device)
    
    print("Label distribution:")
    for i in range(num_classes):
        print(f"  Class {i}: {label_counts[i]:.0f} samples ({100*label_counts[i]/total:.2f}%), weight: {class_weights[i]:.4f}")
    
    return class_weights


def evaluate_model(model, validation_data, feature_dim, device, num_classes):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for page_features, targets in validation_data:
            x = torch.tensor(page_features, dtype=torch.float32).unsqueeze(0).to(device)
            y = torch.tensor(targets, dtype=torch.long)
            
            pred = model(x)
            _, predicted = torch.max(pred.squeeze(0), dim=1)
            
            all_preds.extend(predicted.cpu().tolist())
            all_targets.extend(y.tolist())
    
    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    accuracy = correct / len(all_targets)
    
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    for p, t in zip(all_preds, all_targets):
        class_total[t] += 1
        if p == t:
            class_correct[t] += 1
    
    class_acc = [class_correct[i] / (class_total[i] + 1e-6) for i in range(num_classes)]
    
    return accuracy, class_acc, all_preds, all_targets


if __name__ == "__main__":
    import pickle
    import os
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    num_classes = 11
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    features_cache_file = "features_cache.pkl"
    
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
        dataset = DocLayNetDataset(split="test", rewrite_storage=False)
        if os.path.exists("data_cache.pkl"):
            print("Loading data cache...")
            dataset.load_cache(input_path="data_cache.pkl")
        
        feature_extractor = ImprovedFeatureExtractor(dataset)
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
    training_data = features_and_targets[:int(0.8 * total)]
    validation_data = features_and_targets[int(0.8 * total):]
    print(f"\nTraining samples: {len(training_data)}, Validation samples: {len(validation_data)}")
    
    class_weights = compute_class_weights(pages_targets, num_classes, device)
    model = ImprovedTransformerTagger(
        input_dim=feature_dim,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        output_dim=num_classes,
        dropout=0.2
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    criterion = CombinedLoss(
        num_classes=num_classes,
        alpha=class_weights,
        gamma=2.5,
        label_smoothing=0.1,
        ignore_index=-1
    )
    
    BATCH_SIZE = 16
    EPOCHS = 5
    
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
            torch.save(model.state_dict(), f'best_model-{int(time())}.pth')
    
    print(f"\nTraining completed. Best accuracy: {best_accuracy:.4f}")
