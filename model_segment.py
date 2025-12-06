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
import json
from datetime import datetime
import matplotlib.pyplot as plt
import sys

class FeatureExtractorSegment(FeatureExtractor):
    
    def __init__(self, dataset: DocLayNetDataset | list[PdfFeatures], predictions=None, max_tokens=512):
        """
        Args:
            dataset: DocLayNet dataset or list of PdfFeatures
            predictions: Dict {pdf_idx: {page_idx: [token_predictions]}}
            max_tokens: Maximum tokens per page (must match batch_prelabel.py)
        """
        super().__init__(dataset)
        self.predictions = predictions
        self.max_tokens = max_tokens
    
    def get_page_features(self):
        """Extract features for all pages"""
        pages_features = []
        pages_targets = []
        
        # Determine data source
        data_source = self.dataset if self.dataset else self.pdfs_features
        if data_source is None:
            raise ValueError("No dataset or pdfs_features provided")
        
        pdf_idx = 0
        for pdf_features in tqdm(data_source, desc="Extracting segment features"):
            for page_idx, page in enumerate(pdf_features.pages):
                # Truncate to match batch_prelabel.py
                page.tokens = page.tokens[:self.max_tokens]
                
                # Skip empty pages
                if len(page.tokens) == 0:
                    continue
                
                # If predictions are available, attach them to tokens
                if self.predictions and pdf_idx in self.predictions:
                    if page_idx in self.predictions[pdf_idx]:
                        predictions_list = self.predictions[pdf_idx][page_idx]
                        # Ensure predictions match truncated tokens
                        for i in range(min(len(page.tokens), len(predictions_list))):
                            page.tokens[i]._predicted_type = predictions_list[i]
                
                # Call parent class's loop_token_features to get single page features
                page_feature, page_target = self._extract_page_features(page)
                
                # Double check: only add non-empty pages
                if len(page_feature) > 0:
                    pages_features.append(page_feature)
                    pages_targets.append(page_target)
            
            pdf_idx += 1
        
        return pages_features, pages_targets
    
    def _extract_page_features(self, page):
        """Extract features from a single page"""
        tokens = page.tokens
        if len(tokens) == 0:
            return [], []  
            
        targets = []
        feature_rows = []
        
        for i, token in enumerate(tokens):
            prev_token = tokens[i-1] if i > 0 else None
            next_token = tokens[i+1] if i < len(tokens) - 1 else None
            
            features = self.extract_statistical_features(
                token, page, prev_token, next_token
            )
            
            feature_rows.append(features)
            
            # Segment boundary detection
            # Method 1: Based on token_type change (simple logic)
            if i == 0:
                targets.append(1)  # First token is a boundary
            elif prev_token and hasattr(token, 'token_type') and hasattr(prev_token, 'token_type'):
                # If type changes, consider it a boundary
                if token.token_type != prev_token.token_type:
                    targets.append(1)
                else:
                    targets.append(0)
            else:
                targets.append(0)
        
        return feature_rows, targets
    
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
        
        # font features
        font_id = float(token.font.font_id)
        font_size = float(token.font.font_size)
        features.extend([
            font_id,
            font_size,
            np.log(font_size + 1),
        ])
        
        # positional features
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
        
        # Previous token features (without using token_type)
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
        
        # Next token features (without using token_type)
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
        
        # ALWAYS add prediction features (use -1 as default if no prediction)
        token_type_idx = token._predicted_type if hasattr(token, '_predicted_type') and token._predicted_type is not None else -1
        
        prev_type_idx = -1
        if prev_token and hasattr(prev_token, '_predicted_type') and prev_token._predicted_type is not None:
            prev_type_idx = prev_token._predicted_type
        
        next_type_idx = -1
        if next_token and hasattr(next_token, '_predicted_type') and next_token._predicted_type is not None:
            next_type_idx = next_token._predicted_type
        
        # Additional features (3 dimensions) - ALWAYS ADDED
        additional_features = [
            token_type_idx,                                    
            1 if prev_type_idx == token_type_idx and token_type_idx != -1 else 0,      
            1 if next_type_idx == token_type_idx and token_type_idx != -1 else 0,      
        ]
        
        return features + additional_features
    
def plot_training_history(history, save_path="training_history.png"):
    """draw training history plots"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
        axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(history['train_acc'], label='Train Acc', marker='o')
        axes[0, 1].plot(history['val_acc'], label='Val Acc', marker='s')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning Rate curves
        if 'learning_rate' in history:
            axes[1, 0].plot(history['learning_rate'], marker='o')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].grid(True)
        
        # Segmentation Accuracy (Binary: No Segment/Segment)
        if 'class_acc' in history and len(history['class_acc']) > 0:
            last_class_acc = history['class_acc'][-1]
            class_names = ['No Segment', 'Segment']
            axes[1, 1].bar(class_names, last_class_acc)
            axes[1, 1].set_xlabel('Class')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_title('Segmentation Accuracy (Last Epoch)')
            axes[1, 1].grid(True, axis='y')
        
        # Use constrained_layout instead of tight_layout for NumPy 2.0 compatibility
        fig.set_constrained_layout(True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not save training plot: {e}")
        print("Training completed successfully despite visualization error.")
        # Try a simpler plot
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(history['val_acc'], label='Val Acc', marker='o')
            ax.plot(history['train_acc'], label='Train Acc', marker='s')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Training Progress')
            ax.legend()
            ax.grid(True)
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"Saved simplified training plot to: {save_path}")
        except:
            print("Could not save any training plot.")

def save_training_log(history, args, save_path="training_log.json"):
    """Save training log as JSON"""
    log_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'arguments': vars(args),
        'history': history,
        'best_epoch': history['val_acc'].index(max(history['val_acc'])) + 1,
        'best_val_acc': max(history['val_acc']),
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
    }
    
    with open(save_path, 'w') as f:
        json.dump(log_data, f, indent=2)


class Logger:
    """Logger class that outputs to both terminal and file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', buffering=1)  # line buffering
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()  # 每次写入后立即刷新
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        if not self.log.closed:
            self.log.close()


if __name__ == "__main__":
    import pickle
    import os
    import argparse
    import glob

    parser = argparse.ArgumentParser("DocLayNet Segmentation Training")
    parser.add_argument("--data_path", type=str, help="Path to the DocLayNet dataset")
    parser.add_argument("--predictions", type=str, default=None,
                       help="Path to predictions pickle file from batch_prelabel.py")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"], help="Device to use for training")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument('--split', type=str, default="test", help="Dataset split to use (train/val/test)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--features_cache", type=str, default=".cache/features_cache_segmented.pkl", help="Path to cache extracted features")
    parser.add_argument("--data_cache", type=str, default=".cache/data_cache_segmented.pkl", help="Path to cache dataset")
    parser.add_argument("--train_split", type=float, default=0.8, help="Proportion of data to use for training")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size of the Transformer")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of Transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads in the Transformer")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for optimizer")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focusing parameter for Focal Loss")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--model_save_path", type=str, default="best_model_segment.pth", help="Path to save the best model")
    parser.add_argument("--log_dir", type=str, default="logs_segment", 
                       help="Directory to save logs and plots")
    parser.add_argument("--experiment_name", type=str, default=None, 
                       help="Name for this experiment")
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Maximum tokens per page (must match batch_prelabel.py)")
    args = parser.parse_args()
    
    # Create log directory
    if args.experiment_name:
        log_dir = os.path.join(args.log_dir, args.experiment_name)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(args.log_dir, f"exp_{timestamp}")
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Redirect stdout to both file and terminal
    logger = Logger(os.path.join(log_dir, "training_output.txt"))
    sys.stdout = logger
    
    try:
        print(f"Logging to: {log_dir}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        
        num_classes = 2
        device = args.device
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device)
        
        print(f"Using device: {device}")

        features_cache_file = args.features_cache
        
        # Load predictions
        predictions = None
        if args.predictions and os.path.exists(args.predictions):
            print(f"Loading predictions from {args.predictions}...")
            with open(args.predictions, 'rb') as f:
                predictions = pickle.load(f)
            print(f"Loaded predictions for {len(predictions)} PDFs")
        
        # Check cache
        if os.path.exists(features_cache_file):
            print(f"Found feature cache: {features_cache_file}, Loading cached features...")
            with open(features_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                pages_features = cache_data['pages_features']
                pages_targets = cache_data['pages_targets']
                feature_dim = cache_data['feature_dim']
            print(f"Loaded {len(pages_features)} pages.")
        else:
            print("Loading dataset from DocLayNet...")
            dataset = DocLayNetDataset(split=args.split, rewrite_storage=False)
            if os.path.exists(args.data_cache):
                print("Loading data cache...")
                dataset.load_cache(input_path=args.data_cache)
            
            # Create feature extractor with predictions
            feature_extractor = FeatureExtractorSegment(
                dataset, 
                predictions=predictions,
                max_tokens=args.max_tokens
            )
            
            if not os.path.exists(args.data_cache):
                dataset.save_cache(output_path=args.data_cache)
            
            pages_features, pages_targets = feature_extractor.get_page_features()
            feature_dim = len(pages_features[0][0]) if len(pages_features) > 0 else (39 if not predictions else 42)
            
            # Cache features
            cache_data = {
                'pages_features': pages_features,
                'pages_targets': pages_targets,
                'feature_dim': feature_dim
            }
            print(f"Saving features to cache: {features_cache_file}")
            with open(features_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print("Cache saved.")
        
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
            dropout=args.dropout,
            max_seq_len=args.max_tokens,
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

        # Initialize training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': [],
            'class_acc': [],
            'epoch_time': [],
            'train_class_acc': []
        }
        
        # Training loop
        best_accuracy = 0.0
        
        for epoch in range(args.epochs):
            epoch_start = time()
            model.train()
            total_loss = 0
            all_preds = []
            all_targets = []
            num_batches = 0
            
            random.shuffle(training_data)
            
            for i in tqdm(range(0, len(training_data), args.batch_size), 
                         desc=f"Epoch {epoch+1}/{args.epochs}",
                         file=sys.stdout):
                batch_data = training_data[i:i+args.batch_size]
                batch_x, batch_y, masks, lengths = collate_batch(batch_data, feature_dim, max_seq_len=args.max_tokens)
                
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                logits = model(batch_x, mask=masks)
                
                logits_flat = logits.view(-1, num_classes)
                targets_flat = batch_y.view(-1)

                loss = criterion(logits_flat, targets_flat)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions for training accuracy
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=-1)
                    for pred, target, mask in zip(preds, batch_y, masks):
                        # mask: True=padding (ignore), False=valid (keep)
                        # We want valid tokens, so use ~mask
                        valid_mask = ~mask
                        all_preds.extend(pred[valid_mask].cpu().numpy().tolist())
                        all_targets.extend(target[valid_mask].cpu().numpy().tolist())
            
            avg_train_loss = total_loss / num_batches
            train_acc = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
            
            # Calculate per-class accuracy on training set
            train_class_acc = [0, 0]
            train_class_total = [0, 0]
            for p, t in zip(all_preds, all_targets):
                train_class_total[t] += 1
                if p == t:
                    train_class_acc[t] += 1
            train_class_acc = [train_class_acc[i] / (train_class_total[i] + 1e-6) for i in range(2)]
            
            # Validation
            val_acc, val_loss, val_class_acc = evaluate_model(
                model, validation_data, feature_dim, device, num_classes, max_len=args.max_tokens
            )
            
            # Record learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['learning_rate'].append(current_lr)
            history['class_acc'].append(val_class_acc)
            history['epoch_time'].append(time() - epoch_start)
            history['train_class_acc'].append(train_class_acc)
            
            # Print detailed information
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Epoch Time: {history['epoch_time'][-1]:.2f}s")
            print(f"  Train Per-class: No-Seg={train_class_acc[0]:.3f}, Seg={train_class_acc[1]:.3f}")
            print(f"  Val Per-class: No-Seg={val_class_acc[0]:.3f}, Seg={val_class_acc[1]:.3f}")
            
            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                model_save_path = os.path.join(log_dir, "best_model_segment.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"  ✓ New best model saved! (Val Acc: {val_acc:.4f})")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_accuracy': best_accuracy,
                    'history': history,
                }, checkpoint_path)
                print(f"  Checkpoint saved")
            
            scheduler.step()
        
        # Training completed
        print("\n" + "="*50)
        print("Training completed!")
        print(f"Best validation accuracy: {best_accuracy:.4f}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save training history plot
        plot_path = os.path.join(log_dir, "training_history.png")
        plot_training_history(history, plot_path)
        print(f"Training plot saved to: {plot_path}")
        
        # Save training log
        log_path = os.path.join(log_dir, "training_log.json")
        save_training_log(history, args, log_path)
        if os.path.exists(log_path):
            print(f"✓ Training log saved successfully: {log_path}")
        else:
            print(f"✗ Failed to save training log: {log_path}")
        
        # Save final model
        final_model_path = os.path.join(log_dir, "final_model_segment.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        print(f"\nAll results saved to: {log_dir}")
        
    except Exception as e:
        print(f"\nError occurred during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error log
        error_log_path = os.path.join(log_dir, "error_log.txt")
        with open(error_log_path, 'w') as f:
            f.write(f"Error: {e}\n")
            traceback.print_exc(file=f)
        print(f"Error log saved to: {error_log_path}")
        
    finally:
        # Always close logger
        logger.close()
        sys.stdout = logger.terminal
        print(f"Log file closed: {os.path.join(log_dir, 'training_output.txt')}")
