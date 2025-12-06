import torch
import numpy as np
from model import FeatureExtractor, TransformerTagger
from dataloader import ModifiedPdfFeatures
from reportlab.pdfgen import canvas
from PyPDF2 import PdfReader, PdfWriter
import io
import argparse
from pdf_features.Rectangle import Rectangle
import os
from collections import defaultdict

CATEGORIES = {
    0: "Formula",
    1: "Footnote",
    2: "List item",
    3: "Table",
    4: "Picture",
    5: "Title",
    6: "Text",
    7: "Page header",
    8: "Section header",
    9: "Caption",
    10: "Page footer"
}

def predict_token_types(model, pdf_features, device, max_tokens=512):
    """Predict token types for all pages in PDF"""
    from pdf_token_type_labels.TokenType import TokenType
    
    model.eval()
    
    # Create a single feature extractor for the entire PDF
    feature_extractor = FeatureExtractor(dataset=None)
    feature_extractor.pdfs_features = [pdf_features]
    
    with torch.no_grad():
        for page in pdf_features.pages:
            # Truncate tokens
            original_tokens = page.tokens[:max_tokens]
            page.tokens = original_tokens
            
            if len(page.tokens) == 0:
                continue
            
            # Extract features manually
            features = []
            
            for i, token in enumerate(page.tokens):
                prev_token = page.tokens[i-1] if i > 0 else None
                next_token = page.tokens[i+1] if i < len(page.tokens) - 1 else None
                
                token_features = feature_extractor.extract_statistical_features(
                    token, page, prev_token, next_token
                )
                features.append(token_features)
            
            # Convert to tensor
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Predict
            logits = model(x)
            predictions = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
            # Attach predictions to tokens
            for token, pred in zip(page.tokens, predictions):
                # Set both string label (for visualization) and TokenType object
                pred_int = int(pred)
                token.token_type = TokenType.from_index(pred_int)  # TokenType object
                token._predicted_type = pred_int  # Integer index
                token._predicted_label = CATEGORIES[pred_int]  # String label
    
    return pdf_features


def predict_segments(model, pdf_features, device, max_tokens=512):
    """Predict segment boundaries for all pages in PDF"""
    from model_segment import FeatureExtractorSegment
    
    model.eval()
    
    # Create feature extractor
    feature_extractor = FeatureExtractorSegment(
        dataset=None,
        predictions=None,  # Predictions already attached via _predicted_type
        max_tokens=max_tokens
    )
    feature_extractor.pdfs_features = [pdf_features]
    
    with torch.no_grad():
        for page in pdf_features.pages:
            # Truncate tokens
            original_tokens = page.tokens[:max_tokens]
            page.tokens = original_tokens
            
            if len(page.tokens) == 0:
                continue
            
            # Extract features with predictions
            features = []
            for i, token in enumerate(page.tokens):
                prev_token = page.tokens[i-1] if i > 0 else None
                next_token = page.tokens[i+1] if i < len(page.tokens) - 1 else None
                
                token_features = feature_extractor.extract_statistical_features(
                    token, page, prev_token, next_token
                )
                features.append(token_features)
            
            # Convert to tensor
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Predict
            logits = model(x)
            predictions = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
            # Attach predictions to tokens (1 = segment boundary, 0 = not boundary)
            for token, pred in zip(page.tokens, predictions):
                token.prediction = int(pred)
    
    return pdf_features


def visualize_segments(pdf_features, pdf_path, output_path: str, one_page_only: bool = False):
    """Visualize predicted segments on PDF"""
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    pages = pdf_features.pages 
    
    if len(pages) > len(reader.pages):
        print(f"Warning: {len(pages)} pages in features but only {len(reader.pages)} in PDF. Using PDF page count.")
        pages = pages[:len(reader.pages)]

    for page_idx, page_features in enumerate(pages):
        tokens = page_features.tokens
        page = reader.pages[page_idx]

        packet = io.BytesIO()
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        can = canvas.Canvas(packet, pagesize=(width, height))
        
        # Reset per page
        waitinglist = []
        label_waitinglist = []
        ids = defaultdict(int)

        # Draw segments on page
        for token in tokens:
            # Check if token has prediction attribute
            prediction = getattr(token, 'prediction', 1)  # Default to 1 (boundary) if no segment model
            
            # Use _predicted_label if available (string), otherwise convert TokenType to string
            if hasattr(token, '_predicted_label'):
                token_type = token._predicted_label
            elif hasattr(token, 'token_type'):
                # Convert TokenType object to string
                if hasattr(token.token_type, 'name'):
                    token_type = token.token_type.name
                else:
                    token_type = str(token.token_type)
            else:
                token_type = 'Unknown'
            
            label_waitinglist.append(token_type)

            if prediction == 1:  # Segment boundary
                if waitinglist:  # Only draw if there are tokens accumulated
                    aggregated_rectangle = Rectangle.merge_rectangles(waitinglist + [token.bounding_box])

                    x = float(aggregated_rectangle.left)
                    y = float(aggregated_rectangle.top)
                    w = float(aggregated_rectangle.width)
                    h = float(aggregated_rectangle.height)
                    y_reportlab = height - y - h

                    can.rect(x, y_reportlab, w, h, stroke=1, fill=0)
                    aggregated_label = max(set(label_waitinglist), key=label_waitinglist.count)

                    ids[aggregated_label] += 1
                    segment_id = ids[aggregated_label]
                    can.drawString(x, y_reportlab + h + 5, f"{aggregated_label} {segment_id}")
                
                waitinglist = [token.bounding_box]
                label_waitinglist = [token_type]

            else:
                waitinglist.append(token.bounding_box)

        # Handle last cluster if unfinished
        if waitinglist:
            aggregated_rectangle = Rectangle.merge_rectangles(waitinglist)
            x = float(aggregated_rectangle.left)
            y = float(aggregated_rectangle.top)
            w = float(aggregated_rectangle.width)
            h = float(aggregated_rectangle.height)
            y_reportlab = height - y - h
            can.rect(x, y_reportlab, w, h, stroke=1, fill=0)
            aggregated_label = max(set(label_waitinglist), key=label_waitinglist.count)

            ids[aggregated_label] += 1
            segment_id = ids[aggregated_label]

            can.drawString(x, y_reportlab + h + 5, f"{aggregated_label} {segment_id}")

        can.save()
        packet.seek(0)

        # Merge overlay with original page
        overlay_pdf = PdfReader(packet)
        page.merge_page(overlay_pdf.pages[0])
        writer.add_page(page)
        
        if one_page_only:
            break

    with open(output_path, "wb") as f_out:
        writer.write(f_out)
    
    print(f"Visualization complete! Processed {len(pages) if not one_page_only else 1} page(s).")

def main():
    parser = argparse.ArgumentParser(description="Predict and visualize PDF segments")
    parser.add_argument("--pdf", type=str, required=True, help="Input PDF file path")
    parser.add_argument("--model", type=str, default="best_model.pth", 
                       help="Classification model path")
    parser.add_argument("--output", type=str, default="output.pdf", 
                       help="Output PDF path")
    parser.add_argument("--segment_model", type=str, default=None,
                       help="Segmentation model path (optional)")
    parser.add_argument("--one_page_only", action="store_true",
                       help="Process only first page")
    parser.add_argument("--save_intermediate", type=str, default=None,
                       help="Save intermediate labeled features to pickle file")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Maximum tokens per page")
    
    args = parser.parse_args()
    
    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load PDF
    print(f"Loading PDF from {args.pdf}...")
    pdf_features = ModifiedPdfFeatures.from_pdf_path(args.pdf)
    print(f"Loaded {len(pdf_features.pages)} pages")
    
    # Step 1: Token Classification
    print(f"\nStep 1: Loading classification model from {args.model}...")
    num_classes = 11
    feature_dim = 39
    
    classification_model = TransformerTagger(
        input_dim=feature_dim,
        hidden_dim=256,
        num_layers=8,
        num_heads=8,
        output_dim=num_classes,
        dropout=0.2,
        max_seq_len=args.max_tokens,
    ).to(device)
    
    classification_model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    print("Classification model loaded successfully")
    
    print("Predicting token types...")
    pdf_features = predict_token_types(classification_model, pdf_features, device, args.max_tokens)
    print("Token classification complete")
    
    # Save intermediate results
    if args.save_intermediate:
        import pickle
        with open(args.save_intermediate, 'wb') as f:
            pickle.dump(pdf_features, f)
        print(f"Intermediate features saved to: {args.save_intermediate}")
    
    # Step 2: Segment Prediction (optional)
    if args.segment_model:
        print(f"\nStep 2: Loading segmentation model from {args.segment_model}...")
        segment_feature_dim = 42  # 39 + 3 prediction features
        
        segmentation_model = TransformerTagger(
            input_dim=segment_feature_dim,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            output_dim=2,  # Binary: boundary or not
            dropout=0.2,
            max_seq_len=args.max_tokens,
        ).to(device)
        
        segmentation_model.load_state_dict(torch.load(args.segment_model, map_location=device, weights_only=True))
        print("Segmentation model loaded successfully")
        
        print("Predicting segment boundaries...")
        pdf_features = predict_segments(segmentation_model, pdf_features, device, args.max_tokens)
        print("Segmentation complete")
    else:
        print("\nNo segmentation model provided, using token-level boundaries")
        # Set all tokens as boundaries (each token is a separate segment)
        for page in pdf_features.pages:
            for token in page.tokens:
                token.prediction = 1
    
    # Step 3: Visualization
    print(f"\nStep 3: Generating visualized PDF...")
    visualize_segments(pdf_features, args.pdf, args.output, one_page_only=args.one_page_only)
    print(f"✓ Output saved to: {args.output}")

    
if __name__ == "__main__":
    main()
