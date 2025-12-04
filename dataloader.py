from datasets import load_dataset
import os
import sys
import json
import math
from collections import defaultdict
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType
# features_path = os.path.join(os.getcwd(), "features")
# src_path = os.path.join(os.getcwd(), "src")
# sys.path.append("features")
# if features_path not in sys.path:
#     sys.path.append(features_path)
#     sys.path.append(src_path)

from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.configuration import LABELS_FILE_NAME, TOKEN_TYPE_RELATIVE_PATH, XML_NAME
import subprocess
import tempfile
from os.path import join, exists
from pathlib import Path


TINY_DATA = os.getenv("TINY_DATA", "false").lower() == "true"

class ModifiedPdfFeatures(PdfFeatures):

    @staticmethod
    def from_pdf_path(pdf_path, xml_path: str | Path = None):
        # print(f"Converting...")
        remove_xml = False if xml_path else True
        xml_path = str(xml_path) if xml_path else join(tempfile.gettempdir(), "pdf_etree.xml")

        if PdfFeatures.is_pdf_encrypted(pdf_path):
            subprocess.run(["qpdf", "--decrypt", "--replace-input", pdf_path])

        result = subprocess.run(
            # ["pdftohtml", "-nodrm", "-i", "-xml", "-zoom", "1.0", pdf_path, xml_path],
            ["pdftohtml", "-i", "-xml", "-zoom", "1.0", pdf_path, xml_path],
            # ["pdftohtml", "-xml", pdf_path, xml_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        xml_path += ".xml"
        if not PdfFeatures.contains_text(xml_path):
            subprocess.run(
                ["pdftohtml", "-nodrm", "-i", "-hidden", "-xml", "-zoom", "1.0", pdf_path, xml_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        pdf_features = PdfFeatures.from_poppler_etree(xml_path, file_name=Path(pdf_path).name)

        if remove_xml and exists(xml_path):
            os.remove(xml_path)
        return pdf_features
    
"""
Project annotation labels:
    0: FORMULA = "Formula"
    1: FOOTNOTE = "Footnote"
    2: LIST_ITEM = "List item"
    3: TABLE = "Table"
    4: PICTURE = "Picture"
    5: TITLE = "Title"
    6: TEXT = "Text"
    7: PAGE_HEADER = "Page header"
    8: SECTION_HEADER = "Section header"
    9: CAPTION = "Caption"
    10: PAGE_FOOTER = "Page footer"
"""

"""
Dataset annotation labels:
    1: Caption
    2: Footnote
    3: Formula
    4: List-item
    5: Page-footer
    6: Page-header
    7: Picture
    8: Section-header
    9: Table
    10: Text
    11: Title
"""
dataset_to_project_label_map = {
    1: 9,   # Caption -> Caption
    2: 1,   # Footnote -> Footnote
    3: 0,   # Formula -> Formula
    4: 2,   # List-item -> List item
    5: 10,  # Page-footer -> Page footer
    6: 7,   # Page-header -> Page header
    7: 4,   # Picture -> Picture
    8: 8,   # Section-header -> Section header
    9: 3,   # Table -> Table
    10: 6,  # Text -> Text
    11: 5   # Title -> Title
}

# Remove .xml extension
XML_NAME = XML_NAME[:-4]

class DocLayNetDataset:

    ROOT = "pdf-labeled-data"
    PDF_DIR = 'pdfs'
    ROOT_RELATIVE_FEATURE = ''
    DATASET_NAME = "docling-project/DocLayNet-v1.2" if not TINY_DATA else "SHENJJ1017/TinyDocLayNet"

    def __init__(self, split="train", rewrite_storage=False):
        self.dataset = load_dataset(self.DATASET_NAME, split=split)
        self.split = split
        self.converted = defaultdict(bool)
        self.rewrite_storage = rewrite_storage
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def get_pdf_item(self, idx):
        item = self.dataset[idx]
        pdf_bytes = item["pdf"]
        return pdf_bytes

    def get_pdf_name(self, idx):
        item = self.dataset[idx]
        pdf_name = f"{item['metadata']['original_filename']}_page_{item['metadata']['page_no']}"
        if pdf_name.endswith(".pdf"):
            pdf_name = pdf_name[:-4]
        return pdf_name
    
    def save_pdf_item(self, idx, output_path=None):
        pdf_bytes = self.get_pdf_item(idx)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    def get_item_features(self, idx):
        name = self.get_pdf_name(idx)
        temp_dir = os.path.join(self.ROOT, self.PDF_DIR, name)
        # print('pdf saved to temp dir:', temp_dir)
        # existing files?
        if os.path.exists(temp_dir) and not self.rewrite_storage:
            pdf_feature = ModifiedPdfFeatures.from_pdf_path(os.path.join(temp_dir, "document.pdf"), os.path.join(temp_dir, XML_NAME))
            self.converted[idx] = pdf_feature
            return pdf_feature
        os.makedirs(temp_dir, exist_ok=True)
        pdf_path = os.path.join(temp_dir, f"document.pdf")
        xml_path = os.path.join(temp_dir, XML_NAME)
        self.save_pdf_item(idx, output_path=pdf_path)
        pdf_feature = ModifiedPdfFeatures.from_pdf_path(pdf_path, xml_path)

        label_dir = os.path.join(self.ROOT, TOKEN_TYPE_RELATIVE_PATH, self.split + '_data', name)
        os.makedirs(label_dir, exist_ok=True)
        label_path = os.path.join(label_dir, LABELS_FILE_NAME)
        labels = self.get_labels(idx)
        with open(label_path, "w") as f:
            json.dump(labels, f, indent=4)

        self.converted[idx] = pdf_feature
        return pdf_feature
    
    def get_item_size(self, idx):
        item = self.dataset[idx]
        metadata = item["metadata"]
        original_height, original_width = metadata["original_height"], metadata["original_width"]
        return original_width, original_height

    def get_adjusted_bbox(self, idx):
        item = self.dataset[idx]
        bboxes = item["bboxes"]
        metadata = item["metadata"]
        coco_height, coco_width, original_height, original_width = metadata["coco_height"], metadata["coco_width"], metadata["original_height"], metadata["original_width"]
        scale_x = original_width / coco_width
        scale_y = original_height / coco_height
    
        for i, bb in enumerate(bboxes):
            x, y, w, h = bb
            pdf_x = x * scale_x
            pdf_w_new = w * scale_x
            #pdf_y = original_height - (y + h) * scale_y
            pdf_y = y * scale_y
            pdf_h_new = h * scale_y

            bboxes[i] = [pdf_x, pdf_y, pdf_w_new, pdf_h_new]

        return bboxes
    
    def get_labels(self, idx):
        """
        {
            "pages": [
                {
                    "number": 1,
                    "labels": [
                        {
                            "top": 86,
                            "left": 162,
                            "width": 292,
                            "height": 24,
                            "label_type": 0
                        },
                        {
                            "top": 122,
                            "left": 221,
                            "width": 174,
                            "height": 12,
                            "label_type": 0
                        }
                    ]
                },
                {
                    "number": 2,
                    "labels": [
                        {
                            "top": 36,
                            "left": 296,
                            "width": 22,
                            "height": 13,
                            "label_type": 0
                        },
                        {
                            "top": 72,
                            "left": 71,
                            "width": 473,
                            "height": 49,
                            "label_type": 0
                        }
                    ]
                }
            ]
        }
        """
        adjusted_bboxes = self.get_adjusted_bbox(idx)
        category_ids = self.dataset[idx]["category_id"]
        assert len(adjusted_bboxes) == len(category_ids)
        labels = []
        for bbox, category_id in zip(adjusted_bboxes, category_ids):
            label = {
                "top": math.ceil(bbox[1]),
                "left": math.ceil(bbox[0]),
                "width": math.ceil(bbox[2]),
                "height": math.ceil(bbox[3]),
                "label_type": dataset_to_project_label_map.get(category_id, category_id)
            }
            labels.append(label)
        return {"pages": [{"number": 1, "labels": labels} ]}
    
    def get_segment_labels(self, idx):
        adjusted_bboxes = self.get_adjusted_bbox(idx)
        category_ids = self.dataset[idx]["category_id"]
        assert len(adjusted_bboxes) == len(category_ids)
        segment_labels = []
        features = self.get_item_features(idx)
        bbox_tokens = {
            str(bbox): [token for token in features.pages[0].tokens if token.bounding_box.get_intersection_percentage(Rectangle.from_width_height(left=bbox[0], top=bbox[1], width=bbox[2], height=bbox[3])) > 0] for bbox in adjusted_bboxes
        }
        for bbox in adjusted_bboxes:
            label = {
                "bounding_box": {
                    "top": math.ceil(bbox[1]),
                    "left": math.ceil(bbox[0]),
                    "width": math.ceil(bbox[2]),
                    "height": math.ceil(bbox[3]),
                },
                "label_type": 0
            }
            segment_labels.append(label)
        return {"pages": [{"number": 1, "labels": segment_labels} ]}
    
    def visualize_item(self, idx, output_path=None):
        from reportlab.pdfgen import canvas
        from PyPDF2 import PdfReader, PdfWriter
        import io
        if not self.converted[idx]:
            self.get_item_features(idx)

        label_path = os.path.join(self.ROOT, TOKEN_TYPE_RELATIVE_PATH, self.split + '_data', self.get_pdf_name(idx), LABELS_FILE_NAME)
        pdf_path = os.path.join(self.ROOT, self.PDF_DIR, self.get_pdf_name(idx), "document.pdf")

        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        pdf_w, pdf_h = self.get_item_size(idx)
        with open(label_path, "r") as f:
            labels = json.load(f)
        labels_pages = labels["pages"]
        for i, page in enumerate(reader.pages):
            packet = io.BytesIO()
            can = canvas.Canvas(packet, pagesize=(pdf_w, pdf_h))
            for page_label in labels_pages:
                if page_label["number"] != i + 1:
                    continue
                for label in page_label["labels"]:
                    x = label["left"]
                    y = pdf_h - label["top"] - label["height"]
                    w = label["width"]
                    h = label["height"]
                    c = label["label_type"]
                    can.rect(x, y, w, h, stroke=1, fill=0)
                    can.drawString(x, y + h + 5, str(c))
            can.save()
            packet.seek(0)

            overlay_pdf = PdfReader(packet)
            page.merge_page(overlay_pdf.pages[0])
            writer.add_page(page)

        with open(output_path, "wb") as f_out:
            writer.write(f_out)
    
    def __getitem__(self, idx):
        pdf_name = self.get_pdf_name(idx)
        if self.cache.get(pdf_name, None):
            return self.cache[pdf_name]
        if not self.converted[idx]:
            self.get_item_features(idx)

        features = ModifiedPdfFeatures.from_labeled_data(pdf_labeled_data_root_path = os.path.join(self.ROOT_RELATIVE_FEATURE, self.ROOT), dataset=self.split + '_data', pdf_name=self.get_pdf_name(idx))
        
        adjusted_bboxes = self.get_adjusted_bbox(idx)
        for token in features.pages[0].tokens:
            token.prediction = 0  # reset all tokens to text

        bbox_tokens = {
            str(bbox): [token for token in features.pages[0].tokens if token.bounding_box.get_intersection_percentage(Rectangle.from_width_height(left=bbox[0], top=bbox[1], width=bbox[2], height=bbox[3])) > 0] for bbox in adjusted_bboxes
        }
        for bbox in adjusted_bboxes:
            token_list = bbox_tokens[str(bbox)]
            last_token = token_list[-1] if token_list else None
            if last_token:
                last_token.prediction = 1  # last token defined as the end of segment

        self.cache[pdf_name] = features
        return features

    def save_cache(self, output_path=None):
        import pickle
        cache_path = output_path if output_path else "dataloader_cache.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(self.cache, f)

    def load_cache(self, input_path=None):
        import pickle
        cache_path = input_path if input_path else "dataloader_cache.pkl"
        with open(cache_path, "rb") as f:
            self.cache = pickle.load(f)
            
    def __len__(self):
        return len(self.dataset)


    def visualize_tokens(self, idx, output_path: str, labels: bool = False):
        from reportlab.pdfgen import canvas
        from PyPDF2 import PdfReader, PdfWriter
        import io
        if labels:
            pdf_features = self[idx]
        else:
            pdf_features = self.get_item_features(idx)
        pdf_path = os.path.join(self.ROOT, self.PDF_DIR, self.get_pdf_name(idx), "document.pdf")
        pages = pdf_features.pages
        first_page = pages[0]
        tokens = first_page.tokens

        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        page = reader.pages[0]
        packet = io.BytesIO()
        width, height = self.get_item_size(idx)
        can = canvas.Canvas(packet, pagesize=(width, height))
        id = 0
        for token in tokens:
            id += 1
            bounding_box = token.bounding_box
            l, t, r, b = bounding_box.left, bounding_box.top, bounding_box.right, bounding_box.bottom
            x, y, w, h = l, height - t, r - l, t - b
            token_type = token.token_type


            can.rect(x, y, w, h, stroke=1, fill=0)
            can.drawString(x, y + h + 5, str(token_type) + f"token_{id}")
        can.save()
        packet.seek(0)

        overlay_pdf = PdfReader(packet)
        page.merge_page(overlay_pdf.pages[0])
        writer.add_page(page)

        with open(output_path, "wb") as f_out:
            writer.write(f_out)


    def visualize_segments(self, idx, output_path: str, labels: bool = False):
        from reportlab.pdfgen import canvas
        from PyPDF2 import PdfReader, PdfWriter
        import io
        if labels:
            pdf_features = self[idx]
        else:
            pdf_features = self.__getitem__(idx)
        pdf_path = os.path.join(self.ROOT, self.PDF_DIR, self.get_pdf_name(idx), "document.pdf")
        pages = pdf_features.pages
        first_page = pages[0]
        tokens = first_page.tokens

        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        page = reader.pages[0]
        packet = io.BytesIO()
        width, height = self.get_item_size(idx)
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
                can.drawString(x, y + h + 5, f"segment_{id}")
                waitinglist = []
            else:
                waitinglist.append(token.bounding_box)
        can.save()
        packet.seek(0)

        overlay_pdf = PdfReader(packet)
        page.merge_page(overlay_pdf.pages[0])
        writer.add_page(page)

        with open(output_path, "wb") as f_out:
            writer.write(f_out)

class DocLayNetDatasetSegmented(DocLayNetDataset):

    def __getitem__(self, idx):
        pdf_name = self.get_pdf_name(idx)
        if self.cache.get(pdf_name, None):
            return self.cache[pdf_name]
        if not self.converted[idx]:
            self.get_item_features(idx)

        features = ModifiedPdfFeatures.from_labeled_data(pdf_labeled_data_root_path = os.path.join(self.ROOT_RELATIVE_FEATURE, self.ROOT), dataset=self.split + '_data', pdf_name=self.get_pdf_name(idx))
        adjusted_bboxes = self.get_adjusted_bbox(idx)
        for token in features.pages[0].tokens:
            token.prediction = 0  # reset all tokens to text

        bbox_tokens = {
            str(bbox): [token for token in features.pages[0].tokens if token.bounding_box.get_intersection_percentage(Rectangle.from_width_height(left=bbox[0], top=bbox[1], width=bbox[2], height=bbox[3])) > 0] for bbox in adjusted_bboxes
        }
        for bbox in adjusted_bboxes:
            token_list = bbox_tokens[str(bbox)]
            last_token = token_list[-1] if token_list else None
            if last_token:
                last_token.prediction = 1  # last token defined as the end of segment
        self.cache[pdf_name] = features
        return features
    
    def get_features_segmeted(self, idx):
        features = self.__getitem__(idx)
        # merge tokens into segments based on prediction
        pages_segments = []
        current_segment_tokens = []
        for page in features.pages:
            segments = []
            for token in page.tokens:
                current_segment_tokens.append(token)
                if token.prediction == 1:
                    segments.append(current_segment_tokens)
                    current_segment_tokens = []
            if current_segment_tokens:
                segments.append(current_segment_tokens)
                current_segment_tokens = []
            pages_segments.append(segments)

        pages_segments_positions = []
        for segments in pages_segments:
            segment_positions = []
            for segment_tokens in segments:
                if not segment_tokens:
                    continue
                merged_rectangle = Rectangle.merge_rectangles([token.bounding_box for token in segment_tokens])
                segment_positions.append(merged_rectangle)
            pages_segments_positions.append(segment_positions)   

        return pages_segments_positions


if __name__ == "__main__":
    # It will take 1 hour to download the whole dataset, if you have not done it yet.
    test_dataset = DocLayNetDatasetSegmented(split="test")


    example_idx = 34
    # To see the original labels in the dataset
    features = test_dataset[0]
    print(test_dataset.get_features_segmeted(example_idx))
    #test_dataset.visualize_item(example_idx, output_path="viz_labels.pdf")

    # To see the tokens (automatically extracted) with labels, the token types are all set to "text" since no labels are provided yet.
    #test_dataset.visualize_tokens(example_idx, output_path="token_viz_unlabeled.pdf", labels=False)

    # Combine the labels from the dataset and the tokens extracted, the feature tokens will have the correct labels.
    #features = test_dataset[example_idx]
    #print(features)

    # To see the tokens with correct labels (i.e. viz_labels.pdf + token_viz_unlabeled.pdf = token_viz_labeled.pdf)
    test_dataset.visualize_tokens(example_idx, output_path="token_viz_labeled.pdf", labels=True)