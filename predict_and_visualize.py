import torch
import numpy as np
from model import FeatureExtractor, TransformerTagger
from model_segment import TransformerTaggerSegment
from dataloader import ModifiedPdfFeatures
from reportlab.pdfgen import canvas
from PyPDF2 import PdfReader, PdfWriter
import io
import argparse
from pdf_features.Rectangle import Rectangle
import os

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

def visualize_tokens(pdf_features, pdf_path, output_path: str):
    from reportlab.pdfgen import canvas
    from PyPDF2 import PdfReader, PdfWriter
    import io

    pages = pdf_features.pages
    first_page = pages[0]
    tokens = first_page.tokens

    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    page = reader.pages[0]
    packet = io.BytesIO()
    width, height = page.mediabox.width, page.mediabox.height
    can = canvas.Canvas(packet, pagesize=(width, height))
    ids = [0] * 11
    colors = [
        (1, 0, 0)
    ] * 11 
    
    for i, token in enumerate(tokens):
        bounding_box = token.bounding_box
        l, t, r, b = bounding_box.left, bounding_box.top, bounding_box.right, bounding_box.bottom
        x, y, w, h = l, height - t, r - l, t - b
        
        label = token.token_type.get_index()
        ids[label] += 1
        id = ids[label]
        
        color = colors[label % len(colors)]
        can.setStrokeColorRGB(*color)
        can.setFillColorRGB(*color)
        
        can.rect(x, y, w, h, stroke=1, fill=0)
        
        can.setFont("Helvetica", 12)
        can.drawString(x, y + h + 5, str(token.token_type) + f" {id}")
    can.save()
    packet.seek(0)

    overlay_pdf = PdfReader(packet)
    page.merge_page(overlay_pdf.pages[0])
    writer.add_page(page)

    with open(output_path, "wb") as f_out:
        writer.write(f_out)

def visualize_segments(pdf_features, pdf_path, output_path: str):
    from reportlab.pdfgen import canvas
    from PyPDF2 import PdfReader, PdfWriter
    import io
    
    pages = pdf_features.pages
    first_page = pages[0]
    tokens = first_page.tokens

    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    page = reader.pages[0]
    packet = io.BytesIO()
    width, height = page.mediabox.width, page.mediabox.height
    can = canvas.Canvas(packet, pagesize=(width, height))
    id = 0
    waitinglist = []
    for token in tokens:
        prediction = token.prediction
        if prediction == 1:
            id += 1
            aggregated_rectangle = Rectangle.merge_rectangles(waitinglist + [token.bounding_box])
            x, y, w, h = aggregated_rectangle.left, aggregated_rectangle.top, aggregated_rectangle.width, aggregated_rectangle.height
            y = height - y - h
            can.rect(x, y, w, h, stroke=1, fill=0)
            # can.drawString(x, y + h + 5, f"segment_{id}")
            waitinglist = []
        else:
            waitinglist.append(token.bounding_box)
    if waitinglist:
        id += 1
        aggregated_rectangle = Rectangle.merge_rectangles(waitinglist)
        x, y, w, h = aggregated_rectangle.left, aggregated_rectangle.top, aggregated_rectangle.width, aggregated_rectangle.height
        y = height - y - h
        can.rect(x, y, w, h, stroke=1, fill=0)
    
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
    
    pdf_features = ModifiedPdfFeatures.from_pdf_path(args.pdf)
    
    feature_dim = 39

    print(f"Loading model from {args.model}...")
    num_classes = 11
    model = TransformerTagger(
        input_dim=feature_dim,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        output_dim=num_classes,
        dropout=0.2,
        pdfs_features= [pdf_features]
    ).to(device)
    
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f" Model loaded successfully")
    # predicted_labels = predict(model, features, device)
    pdf_features = model.labeled_features()[0]

    model_segment = TransformerTaggerSegment(
        input_dim=feature_dim + 3,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        output_dim=2,
        dropout=0.2,
        pdfs_features= [pdf_features]
    ).to(device)
    model_segment.load_state_dict(torch.load("best_model-1764799978_segmented.pth", map_location=device))
    pdf_features = model_segment.labeled_features()[0]
    visualize_segments(pdf_features, args.pdf, args.output)
    print(f"Output saved to: {args.output}")

    
if __name__ == "__main__":
    main()
