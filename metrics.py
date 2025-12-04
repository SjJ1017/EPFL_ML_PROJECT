from pdf_features.Rectangle import Rectangle
from dataloader import ModifiedPdfFeatures, DocLayNetDataset, DocLayNetDatasetSegmented
import torch
from model import TransformerTagger
from model_segment import TransformerTaggerSegment
from predict_and_visualize import visualize_segments

def miou_page(rectangles_a, rectangles_b, labels_a, labels_b, num_classes):
    """
    Compute mean Intersection over Union (mIoU) between two sets of rectangles with labels.
    rectangles_a: List of Rectangle objects for set A
    rectangles_b: List of Rectangle objects for set B # gold
    labels_a: List of integer labels corresponding to rectangles in set A
    labels_b: List of integer labels corresponding to rectangles in set B
    num_classes: Total number of classes (labels)
    """
    iou_per_class = [0.0] * num_classes
    count_per_class = [0] * num_classes

    intersection_per_class = [0.0] * num_classes
    union_per_class = [0.0] * num_classes

    for i in range(num_classes):
        rects_a_class = [rect for rect, label in zip(rectangles_a, labels_a) if label == i]
        rects_b_class = [rect for rect, label in zip(rectangles_b, labels_b) if label == i]
        for rect_b in rects_b_class:
            union_area = rect_b.area()
            intersection_area = 0.0
            for rect_a in rects_a_class:
                inter = rect_a.get_intersection_percentage(rect_b) * rect_a.area() / 100
                #print(f"Class {i}: Intersection between {rect_a.to_dict()} and {rect_b.to_dict()} is {inter}")
            if inter > 0:
                intersection_area += inter
                
                union_area += rect_a.area() - intersection_area
            else:
                union_area += rect_a.area()
            intersection_per_class[i] += intersection_area
            #print(f"Class {i}: Total Intersection Area: {intersection_per_class[i]}, Total Union Area: {union_per_class[i]}")
            union_per_class[i] += union_area
        if union_per_class[i] > 0:
            iou_per_class[i] = intersection_per_class[i] / union_per_class[i]
            count_per_class[i] += 1

    total_intersection = sum(intersection_per_class)
    total_union = sum(union_per_class)
    mean_iou = total_intersection / total_union if total_union > 0 else 0.0
    return {
        'micro_iou': mean_iou,
        'macro_iou': sum(iou_per_class) / num_classes,
        'iou_per_class': iou_per_class
    }

def get_rectangles_from_page(page):
    segments = []
    current_segment_tokens = []
    for token in page.tokens:
        current_segment_tokens.append(token)
        if token.prediction == 1:
            segments.append(current_segment_tokens)
            current_segment_tokens = []
    if current_segment_tokens:
        segments.append(current_segment_tokens)
    
    rectangles = []
    labels = []
    for segment in segments:
        token_rectangles = [token.bounding_box for token in segment]
        merged_rectangle = Rectangle.merge_rectangles(token_rectangles)
        rectangles.append(merged_rectangle)
        all_token_types = [token.token_type.get_index() for token in segment]
        label = max(set(all_token_types), key=all_token_types.count)
        labels.append(label)

    return rectangles, labels


def evaluate(features_a, features_b, num_classes):
    pages_rectangles_a = []
    pages_rectangles_b = []
    pages_labels_a = []
    pages_labels_b = []

    for pdf_feature_a, pdf_feature_b in zip(features_a, features_b):
        for page_a, page_b in zip(pdf_feature_a.pages, pdf_feature_b.pages):
            rectangles_a, page_labels_a = get_rectangles_from_page(page_a)
            rectangles_b, page_labels_b = get_rectangles_from_page(page_b)
            pages_rectangles_a.append(rectangles_a)
            pages_rectangles_b.append(rectangles_b)
            pages_labels_a.append(page_labels_a)
            pages_labels_b.append(page_labels_b)
    all_micro_ious = []
    all_macro_ious = []
    for rects_a, rects_b, labels_a, labels_b in zip(pages_rectangles_a, pages_rectangles_b, pages_labels_a, pages_labels_b):
        iou_result = miou_page(rects_a, rects_b, labels_a, labels_b, num_classes)
        all_micro_ious.append(iou_result['micro_iou'])
        all_macro_ious.append(iou_result['macro_iou'])
    overall_micro_iou = sum(all_micro_ious) / len(all_micro_ious)
    overall_macro_iou = sum(all_macro_ious) / len(all_macro_ious)
    ious_per_class = {}
    for i in range(num_classes):
        class_ious = []
        for rects_a, rects_b, labels_a, labels_b in zip(pages_rectangles_a, pages_rectangles_b, pages_labels_a, pages_labels_b):
            iou_result = miou_page(rects_a, rects_b, labels_a, labels_b, num_classes)
            class_ious.append(iou_result['iou_per_class'][i])
        ious_per_class[i] = sum(class_ious) / len(class_ious)
        
    return {
        'micro_iou': overall_micro_iou,
        'macro_iou': overall_macro_iou,
        "ious_per_class": ious_per_class
    }


# Test
if __name__ == "__main__":
    """rect_a1 = Rectangle.from_coordinates(0, 0, 5, 2)
    rect_a2 = Rectangle.from_coordinates(2, 1, 4, 4)
    rect_b1 = Rectangle.from_coordinates(0, 0, 10, 2)
    rect_b2 = Rectangle.from_coordinates(2, 2, 5, 4)
    rectangles_a = [rect_a1, rect_a2]
    rectangles_b = [rect_b1, rect_b2]
    labels_a = [0, 1]
    labels_b = [0, 1]
    num_classes = 2
    miou_score = miou_page(rectangles_a, rectangles_b, labels_a, labels_b, num_classes)
    print(f"mIoU Score: {miou_score}")  # Expected output: mIoU Score: 0.5"""
    test_dataset = DocLayNetDatasetSegmented(split="test")
    pdfs_features = [test_dataset[i] for i in range(10)]
    
    feature_dim = 39

    num_classes = 11
    model = TransformerTagger(
        input_dim=feature_dim,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        output_dim=num_classes,
        dropout=0.2,
        pdfs_features= pdfs_features
    )
    
    model.load_state_dict(torch.load("best_model.pth"))

    type_pdfs_features = model.labeled_features()

    model_segment = TransformerTaggerSegment(
        input_dim=feature_dim + 3,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        output_dim=2,
        dropout=0.2,
        pdfs_features= type_pdfs_features
    )
    model_segment.load_state_dict(torch.load("best_model-1764799978_segmented.pth"))
    segmented_pdf_features = model_segment.labeled_features()
    evaluation_result = evaluate(type_pdfs_features, segmented_pdf_features, num_classes)
    print("Evaluation Result:", evaluation_result)

    # first pdf path
    import os

    for i in range(10):
        pdf_path = os.path.join(test_dataset.ROOT, test_dataset.PDF_DIR, test_dataset.get_pdf_name(i), "document.pdf")
        visualize_segments(segmented_pdf_features[i], pdf_path, f"test_pdf_labeled/test_output_{i}.pdf")

    