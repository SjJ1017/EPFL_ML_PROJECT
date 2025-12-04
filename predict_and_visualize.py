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

def visualize_segments(pdf_features, pdf_path, output_path: str, one_page_only: bool = False):
    from reportlab.pdfgen import canvas
    from PyPDF2 import PdfReader, PdfWriter
    import io
    from collections import defaultdict
    
    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    pages = pdf_features.pages 
    assert len(pages) == len(reader.pages), "Number of pages in features and PDF do not match."

    for page_idx, page_features in enumerate(pages):
        tokens = page_features.tokens
        page = reader.pages[page_idx]

        packet = io.BytesIO()
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        can = canvas.Canvas(packet, pagesize=(width, height))
        
        # reset per page
        waitinglist = []
        label_waitinglist = []
        ids = defaultdict(int)

        # draw segments on page
        for token in tokens:
            prediction = token.prediction
            label_waitinglist.append(token.token_type)

            if prediction == 1:
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
                waitinglist, label_waitinglist = [], []

            else:
                waitinglist.append(token.bounding_box)

        # handle last cluster if unfinished
        if waitinglist:
            aggregated_rectangle = Rectangle.merge_rectangles(waitinglist)
            x = float(aggregated_rectangle.left)
            y = float(aggregated_rectangle.top)
            w = float(aggregated_rectangle.width)
            h = float(aggregated_rectangle.height)
            y_reportlab = height - y - h
            can.rect(x, y_reportlab, w, h, stroke=1, fill=0)


        can.save()
        packet.seek(0)

        # merge overlay with original page
        overlay_pdf = PdfReader(packet)
        page.merge_page(overlay_pdf.pages[0])
        writer.add_page(page)
        if one_page_only:
            break

    with open(output_path, "wb") as f_out:
        writer.write(f_out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, required=True)
    parser.add_argument("--model", type=str, default="best_model.pth")
    parser.add_argument("--output", type=str, default="output.pdf")
    parser.add_argument("--segment_model", type=str, default=None)
    parser.add_argument("--one_page_only", action="store_true")
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
    model_segment.load_state_dict(torch.load(args.segment_model, map_location=device))
    pdf_features = model_segment.labeled_features()[0]
    visualize_segments(pdf_features, args.pdf, args.output, one_page_only=args.one_page_only)
    print(f"Output saved to: {args.output}")

    
if __name__ == "__main__":
    main()
