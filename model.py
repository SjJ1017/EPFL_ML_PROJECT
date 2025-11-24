from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.PdfToken import PdfToken
from tqdm import tqdm
import numpy as np
from pathlib import Path
from dataloader import DocLayNetDataset
from transformers import AutoTokenizer
import torch.optim as optim


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

import torch


class FeatureExtractor:

    max_length = 14
    

    def __init__(self, dataset: DocLayNetDataset):
        self.dataset = dataset

    def get_page_features(self):
        # stop at k
        k = float('inf')
        start = 0
        pages_features = []
        page_targets = []
        for page in self.loop_token_features():
            targets = []
            feature_rows = []
            features = page
            for token in page.tokens:
                # token ids, padding to max length
                content = token.content
                encoding = tokenizer(content, add_special_tokens=True, return_tensors="pt")
                token_ids = encoding["input_ids"][0].tolist()
                if len(token_ids) < self.max_length:
                    padding_length = self.max_length - len(token_ids)
                    token_ids += [tokenizer.pad_token_id] * padding_length
                else:
                    token_ids = token_ids[:self.max_length]
                
                # font info
                font_id = float(token.font.font_id)
                font_size = float(token.font.font_size)
                font_features = [font_id, font_size]

                # position info
                box = token.bounding_box
                positions = [box.left, box.top, box.right, box.bottom]

                features = token_ids + font_features + positions
                feature_rows.append(features)
                targets.append(token.token_type.get_index())
            pages_features.append(feature_rows)
            page_targets.append(targets)
            start += 1
            if start >= k:
                break

        return pages_features, page_targets

    def loop_token_features(self):
        for pdf_features in tqdm(self.dataset):
            for page in pdf_features.pages:
                if not page.tokens:
                    continue

                yield page


import torch.nn as nn
from transformers import BertConfig, BertModel



class Seq2SeqTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_heads=4, output_dim=1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.pos_embedding = nn.Parameter(torch.randn(5000, hidden_dim))  
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, target=None):
        """
        x: (N, M) -> reshape to (M, N) for sequence first
        target: not used in forward pass, only for training
        """
        
        M = x.shape[1] # (M, batch, hidden_dim)
        x = x.T  # (M, N)
        x_emb = self.input_proj(x) + self.pos_embedding[:M]  # (M, hidden_dim)
        x_emb = x_emb.unsqueeze(1)  # (M, 1, hidden_dim)
        
        memory = self.encoder(x_emb)  # (M, 1, hidden_dim)
        
        out = memory.squeeze(1)  # (M, hidden_dim)
        
        out = self.output_proj(out)  # (M, output_dim)
        return out



if __name__ == "__main__":
    N = 20
    num_classes = 11

    dataset = DocLayNetDataset(split="test", rewrite_storage=False)
    feature_extractor = FeatureExtractor(dataset)
    pages_features, pages_targets = feature_extractor.get_page_features()
    model = Seq2SeqTransformer(input_dim=N, output_dim=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    features_and_targets = list(zip(pages_features, pages_targets))
    import random
    random.shuffle(features_and_targets)
    total = len(features_and_targets)
    training_data = features_and_targets[:int(0.8 * total)]
    validation_data = features_and_targets[int(0.8 * total):]
    print(f"Training samples: {len(training_data)}, Validation samples: {len(validation_data)}")

    EPOCHS = 1
    losses = []
    accuracies = []
    for epoch in range(EPOCHS):
        for i, (page_features, targets) in enumerate(training_data):
            x, y = page_features, targets
            x = torch.tensor(x, dtype=torch.float32).reshape(N, -1)
            y = torch.tensor(y, dtype=torch.long)

            optimizer.zero_grad()
            pred = model(x) 
            loss = criterion(pred, y)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # accuracy
            if (i + 1) % 500 == 0:
                for page_features_val, targets_val in validation_data:
                    x_val = torch.tensor(page_features_val, dtype=torch.float32).reshape(N, -1)
                    y_val = torch.tensor(targets_val, dtype=torch.long)
                    with torch.no_grad():
                        pred_val = model(x_val)
                        _, predicted_val = torch.max(pred_val, dim=1)
                        correct_val = (predicted_val == y_val).sum().item()
                        accuracy = correct_val / y_val.size(0)

                print(f"Iteration {i+1} | Validation Accuracy = {accuracy:.4f}")
        print(f"Epoch {epoch} | Loss = {loss.item():.4f}, Accuracy = {accuracy:.4f}")

    import matplotlib.pyplot as plt

    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()