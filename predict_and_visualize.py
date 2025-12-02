import torch
import numpy as np
from model import ImprovedFeatureExtractor, ImprovedTransformerTagger
from dataloader import ModifiedPdfFeatures
from reportlab.pdfgen import canvas
from PyPDF2 import PdfReader, PdfWriter
import io
import argparse

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


def extract_single_page_features(pdf_path):
    print(f"Loading PDF: {pdf_path}")
    pdf_features = ModifiedPdfFeatures.from_pdf_path(pdf_path)
    page = pdf_features.pages[0]
    tokens = page.tokens
    
    if len(tokens) == 0:
        print("Warning: No tokens found in the PDF!")
        return None, None, None
    
    print(f"Found {len(tokens)} tokens")
    
    feature_rows = []
    for i, token in enumerate(tokens):
        prev_token = tokens[i-1] if i > 0 else None
        next_token = tokens[i+1] if i < len(tokens) - 1 else None
        
        features = extract_token_features(token, page, prev_token, next_token)
        feature_rows.append(features)
    
    return feature_rows, tokens, pdf_features


def extract_token_features(token, page, prev_token=None, next_token=None):
    features = []
    
    content = token.content
    features.extend([
        len(content),  
        len(content.split()),  
        sum(1 for c in content if c.isupper()) / (len(content) + 1e-6),  
        sum(1 for c in content if c.isdigit()) / (len(content) + 1e-6), 
        sum(1 for c in content if c.isspace()) / (len(content) + 1e-6),
        1.0 if content.isupper() else 0.0,
        1.0 if (content and content[0].isupper()) else 0.0,  
        1.0 if any(c.isdigit() for c in content) else 0.0, 
        1.0 if any(c in '.,;:!?' for c in content) else 0.0,
    ])
    
    font_id = float(token.font.font_id)
    font_size = float(token.font.font_size)
    features.extend([
        font_id,
        font_size,
        np.log(font_size + 1),
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


def predict(model, features, device):
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, feature_dim)
    
    model.eval()
    with torch.no_grad():
        predictions = model(x)  
        _, predicted_labels = torch.max(predictions.squeeze(0), dim=1)
    
    return predicted_labels.cpu().tolist()


def visualize_tokens(pdf_features, pdf_path, output_path, predicted_labels):
    pages = pdf_features.pages
    first_page = pages[0]
    tokens = first_page.tokens

    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    page = reader.pages[0]
    packet = io.BytesIO()
    width, height = page.mediabox.width, page.mediabox.height
    width, height = float(width), float(height)
    
    can = canvas.Canvas(packet, pagesize=(width, height))
    
    colors = [
        (1, 0, 0)
    ] * 11 
    
    for i, token in enumerate(tokens):
        bounding_box = token.bounding_box
        l, t, r, b = bounding_box.left, bounding_box.top, bounding_box.right, bounding_box.bottom
        x, y, w, h = l, height - t, r - l, t - b
        
        label = predicted_labels[i]
        category_name = CATEGORIES.get(label, f"Class_{label}")
        
        color = colors[label % len(colors)]
        can.setStrokeColorRGB(*color)
        can.setFillColorRGB(*color)
        
        can.rect(x, y, w, h, stroke=1, fill=0)
        
        can.setFont("Helvetica", 12)
        can.drawString(x, y + h + 2, category_name)
    
    can.save()
    packet.seek(0)

    overlay_pdf = PdfReader(packet)
    page.merge_page(overlay_pdf.pages[0])
    writer.add_page(page)

    with open(output_path, "wb") as f_out:
        writer.write(f_out)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, required=True)
    parser.add_argument("--model", type=str, default="best_model.pth")
    parser.add_argument("--output", type=str, default="output.pdf")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "mps"])
    
    args = parser.parse_args()
    
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
    
    features, tokens, pdf_features = extract_single_page_features(args.pdf)
    
    if features is None:
        print("Failed to extract features. Exiting.")
        return
    
    feature_dim = len(features[0])

    print(f"Loading model from {args.model}...")
    num_classes = 11
    model = ImprovedTransformerTagger(
        input_dim=feature_dim,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        output_dim=num_classes,
        dropout=0.2
    ).to(device)
    
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f" Model loaded successfully")
    predicted_labels = predict(model, features, device)

    visualize_tokens(pdf_features, args.pdf, args.output, predicted_labels)
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
