import torch
import os
import argparse
from tqdm import tqdm
from model import TransformerTagger
from dataloader import DocLayNetDataset
import pickle
import numpy as np

def main():
    parser = argparse.ArgumentParser("Batch Prelabeling for Segment Model Training")
    parser.add_argument("--model", type=str, required=True, 
                       help="Path to trained classification model (.pth)")
    parser.add_argument("--split", type=str, default="train", 
                       choices=["train", "val", "test"],
                       help="Dataset split to prelabel")
    parser.add_argument("--output_file", type=str, required=True, 
                       help="Output file for predictions (e.g., predictions_train.pkl)")
    parser.add_argument("--data_cache", type=str, default=".cache/data_cache.pkl",
                       help="Path to data cache")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max_seq_len", type=int, default=512,
                       help="Max sequence length (must match training)")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--batch_process", type=int, default=10,
                       help="Process N PDFs at a time")
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create cache directory
    cache_dir = os.path.dirname(args.data_cache)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Created cache directory: {cache_dir}")
    
    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading {args.split} dataset...")
    dataset = DocLayNetDataset(split=args.split, rewrite_storage=False)
    if os.path.exists(args.data_cache):
        dataset.load_cache(args.data_cache)
    else:
        dataset.save_cache(args.data_cache)
    
    print(f"Total PDFs to process: {len(dataset)}")
    
    # Load model weights
    print(f"Loading model from {args.model}...")
    feature_dim = 39
    num_classes = 11
    
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    # Store all predictions: {pdf_idx: {page_idx: [token_predictions]}}
    all_predictions = {}
    
    # Batch processing
    for batch_start in tqdm(range(0, len(dataset), args.batch_process), 
                            desc="Processing batches"):
        batch_end = min(batch_start + args.batch_process, len(dataset))
        batch_pdfs = [dataset[i] for i in range(batch_start, batch_end)]
        
        # Limit tokens per PDF
        for pdf in batch_pdfs:
            for page in pdf.pages:
                page.tokens = page.tokens[:args.max_seq_len]
        
        # Create model for this batch
        model = TransformerTagger(
            input_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            output_dim=num_classes,
            dropout=0.2,
            pdfs_features=batch_pdfs,
            max_seq_len=args.max_seq_len,
        ).to(device)
        
        model.load_state_dict(checkpoint)
        model.eval()
        
        # Get predictions (flat list of all tokens)
        all_token_predictions = model.predict()
        
        # Organize predictions by PDF and page
        token_idx = 0
        for i, pdf in enumerate(batch_pdfs):
            pdf_idx = batch_start + i
            all_predictions[pdf_idx] = {}
            
            for p_idx, page in enumerate(pdf.pages):
                num_tokens = len(page.tokens)
                # Extract predictions for this page
                page_preds = all_token_predictions[token_idx:token_idx + num_tokens]
                all_predictions[pdf_idx][p_idx] = page_preds
                token_idx += num_tokens
        
        # Clear memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save predictions
    print(f"\nSaving predictions to {args.output_file}...")
    with open(args.output_file, 'wb') as f:
        pickle.dump(all_predictions, f)
    
    # Calculate file size
    file_size_mb = os.path.getsize(args.output_file) / (1024 * 1024)
    
    print(f"\nPrelabeling complete!")
    print(f"Saved predictions for {len(dataset)} PDFs")
    print(f"File size: {file_size_mb:.2f} MB (vs ~50 GB for full PDFs)")
    print(f"\nNext step: Train segment model with:")
    print(f"  python model_segment.py --predictions {args.output_file} --split {args.split}")

if __name__ == "__main__":
    main()