from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.PdfToken import PdfToken
from pdf_features.PdfPage import PdfPage
from tqdm import tqdm
import numpy as np
from pathlib import Path
from dataloader import DocLayNetDataset
from transformers import AutoTokenizer
import torch.optim as optim


# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

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
            page_width = page.page_width
            page_height = page.page_height
            targets = []
            feature_rows = []
            features = page
            for token in page.tokens:
                # token ids, padding to max length
                """content = token.content
                encoding = tokenizer(content, add_special_tokens=True, return_tensors="pt")
                token_ids = encoding["input_ids"][0].tolist()
                if len(token_ids) < self.max_length:
                    padding_length = self.max_length - len(token_ids)
                    token_ids += [tokenizer.pad_token_id] * padding_length
                else:
                    token_ids = token_ids[:self.max_length]"""
                content = token.content
                content_len = len(content)
                token_ids = [content_len]

                # font info
                font_id = float(token.font.font_id)
                font_size = float(token.font.font_size)
                font_features = [font_id, font_size]

                # position info
                box = token.bounding_box
                positions = [box.left, box.top, box.right, box.bottom, (box.top - box.bottom)/page_height, (box.right-box.left)/page_width]

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
        
        self.pos_embedding = nn.Parameter(torch.randn(50, hidden_dim))  
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Transformer Decoder
        # decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        # self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, target=None):
        """
        x: (batch_size, N, M) 或 (N, M) for single sample
        target: not used in forward pass, only for training
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, N, M)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, N, M = x.shape
        
        # (batch_size, N, M) -> (batch_size, M, N) -> (M, batch_size, N)
        x = x.transpose(1, 2).transpose(0, 1)  # (M, batch_size, N)
        x_emb = self.input_proj(x) + self.pos_embedding[:M].unsqueeze(1)  # (M, batch_size, hidden_dim)
        
        memory = self.encoder(x_emb)  # (M, batch_size, hidden_dim)
        
        # (M, batch_size, hidden_dim) -> (batch_size, M, hidden_dim)
        out = memory.transpose(0, 1)
        
        out = self.output_proj(out)  # (batch_size, M, output_dim)
        
        if squeeze_output:
            out = out.squeeze(0)  # (M, output_dim)
        
        return out



def collate_batch(batch_data, N):
    batch_x = []
    batch_y = []
    lengths = []
    
    for page_features, targets in batch_data:
        x = torch.tensor(page_features, dtype=torch.float32).reshape(N, -1)
        y = torch.tensor(targets, dtype=torch.long)
        batch_x.append(x)
        batch_y.append(y)
        lengths.append(len(targets))
    
    # Padding to same length
    max_len = max(lengths)
    padded_x = []
    padded_y = []
    
    for x, y, length in zip(batch_x, batch_y, lengths):
        if x.shape[1] < max_len:
            # Pad x
            pad_size = max_len - x.shape[1]
            x_padded = torch.cat([x, torch.zeros(N, pad_size)], dim=1)
            # Pad y with -1 (ignore index)
            y_padded = torch.cat([y, torch.full((pad_size,), -1, dtype=torch.long)])
        else:
            x_padded = x
            y_padded = y
        padded_x.append(x_padded)
        padded_y.append(y_padded)
    
    batch_x = torch.stack(padded_x)  # (batch_size, N, max_len)
    batch_y = torch.stack(padded_y)  # (batch_size, max_len)
    
    return batch_x, batch_y, lengths


if __name__ == "__main__":
    N = 20
    num_classes = 11

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    print("Using device:", device)
    dataset = DocLayNetDataset(split="test", rewrite_storage=False)
    dataset.load_cache(input_path="data_cache.pkl")
    print("Dataset cache loaded.")
    # Done
    feature_extractor = FeatureExtractor(dataset)
    pages_features, pages_targets = feature_extractor.get_page_features()


    model = Seq2SeqTransformer(input_dim=N, output_dim=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss().to(device)

    features_and_targets = list(zip(pages_features, pages_targets))
    import random
    random.shuffle(features_and_targets)
    total = len(features_and_targets)
    training_data = features_and_targets[:int(0.8 * total)]
    validation_data = features_and_targets[int(0.8 * total):]
    print(f"Training samples: {len(training_data)}, Validation samples: {len(validation_data)}")


    BATCH_SIZE = 64
    EPOCHS = 1
    losses = []
    accuracies = []
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    for epoch in range(EPOCHS):
        j = 0
        for i in tqdm(range(0, len(training_data), BATCH_SIZE)):
            j += 1
            batch_data = training_data[i:i+BATCH_SIZE]
            
            x, y, lengths = collate_batch(batch_data, N)
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)  # (batch_size, M, num_classes)
            
            pred_flat = pred.view(-1, num_classes)  # (batch_size * M, num_classes)
            y_flat = y.view(-1)  # (batch_size * M)
            
            loss = criterion(pred_flat, y_flat)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


    import matplotlib.pyplot as plt

    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()