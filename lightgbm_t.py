import sys
import os
import time

from pdf_features.PdfFeatures import PdfFeatures
# pdf-labeled-data/labeled_data/token_type/test_data
import os
from tqdm import tqdm


def train_tyken_type():

    all_files = os.listdir("../ml/pdf-labeled-data/labeled_data/token_type/test_data")[:100]
    pdfs_features = []
    for file in tqdm(all_files):
        pdf_feature = PdfFeatures.from_labeled_data("../ml/pdf-labeled-data", "test_data", file)
        pdfs_features.append(pdf_feature)


    features_path = os.path.join(os.getcwd(), "features")
    src_path = os.path.join(os.getcwd(), "src")

    if features_path not in sys.path:
        sys.path.append(features_path)
        sys.path.append(src_path)


    from pdf_features.PdfFeatures import PdfFeatures
    from src.adapters.ml.pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
    from src.adapters.ml.pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer


    def train_token_type_model():
        model_configuration = ModelConfiguration()
        labeled_pdf_features_list: list[PdfFeatures] = pdfs_features[:100]
        trainer = TokenTypeTrainer(labeled_pdf_features_list, model_configuration)
        train_labels = [token.token_type.get_index() for token in trainer.loop_tokens()]
        start_time = time.time()
        trainer.train("models/token_type_example_model__.model", train_labels)    
        end_time = time.time()
        print(f"Training time: {end_time - start_time} seconds")
    train_token_type_model()

def train_segmenter():
    from dataloader import DocLayNetDatasetSegmented
    dataset = DocLayNetDatasetSegmented(split='test')

    pdfs_features = []
    for i in range(len(dataset)):
        pdfs_features.append(dataset[i])

    print([tok.prediction for tok in pdfs_features[0].pages[0].tokens])

    features_path = os.path.join(os.getcwd(), "features")
    src_path = os.path.join(os.getcwd(), "src")

    if features_path not in sys.path:
        sys.path.append(features_path)
        sys.path.append(src_path)


    from pdf_features.PdfFeatures import PdfFeatures
    from src.adapters.ml.pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
    from src.adapters.ml.fast_trainer.ParagraphExtractorTrainer import ParagraphExtractorTrainer


    def train_segmenter_model():
        model_configuration = ModelConfiguration()
        labeled_pdf_features_list: list[PdfFeatures] = pdfs_features
        trainer = ParagraphExtractorTrainer(labeled_pdf_features_list, model_configuration)
        train_labels = [token.prediction for token in trainer.loop_tokens()]
        start_time = time.time()
        trainer.train("models/paragraph_extractor_example_model_.model", train_labels)
        end_time = time.time()
        print(f"Training time: {end_time - start_time} seconds")
    train_segmenter_model()

train_segmenter()