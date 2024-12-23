import os
import sys

sys.path.append(os.path.join("..", os.getcwd()))

import hydra
import pandas as pd
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.dataset import CSVDataset
from src.multihead_attn import TransformerEncoder
from torch.utils.data import DataLoader

from src.utils import remove_duplicate_strings

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# csv_file= "generated_claim_triplets_with_topics.csv"
# csv_file_path = os.path.join("data", csv_file)
# data = pd.read_csv(csv_file_path)

bi_encoder_model_name = "pritamdeka/PubMedBERT-mnli-snli-scinli-stsb"
cross_encoder_model_path = "/content/drive/MyDrive/fine_tuned_cross_encoder"

class Retriever(nn.Module):
    """Given a list of evidences and a claim, this returns the top-k evidences"""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.bi_encoder = SentenceTransformer(cfg.bi_encoder_model_name, cache_folder="../cache")
        self.k = cfg.k
        self.data = pd.read_csv(cfg.csv_file)
        self.evidence_pool = remove_duplicate_strings(self.data["Evidence"].dropna().tolist())
        self.evidence_embeddings = self.bi_encoder.encode(self.evidence_pool, convert_to_tensor=True)
        
    def tokenize_and_embed(self, data):
        # data -> [b]
        return self.bi_encoder.encode([data], convert_to_tensor=True)
    
    def set_encoder_training(self, mode):
        self.bi_encoder.train(mode)
    
    def forward(self, x):
        # x -> b, claims
        x = self.bi_encoder.encode(x, convert_to_tensor=True)
        # scores -> b, num_evidences, each row is the cosine similarity b/w the claim
        # and all the evidences
        cos_sim = torch.mm(x, self.evidence_embeddings.T)
        scores, indices = torch.topk(cos_sim, self.k, dim=1)
        evidences = [[self.evidence_pool[i] for i in row] for row in indices]
        return scores, indices, evidences
        
class Ranker(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        # self.cross_encoder = AutoModelForSequenceClassification.from_pretrained(cfg.cross_encoder_model_name, cache_dir="../cache")
        self.cross_encoder = SentenceTransformer(cfg.bi_encoder_model_name, cache_folder="../cache")
        # self.cross_encoder.classifier = nn.Identity()  # remove the last classifier layer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.cross_encoder_model_name)
        hidden_size = self.cross_encoder.config.hidden_size
        self.scorer = nn.Linear(hidden_size, 1)
        
    def forward(self, x, evidence_pool):
        # x -> b, claim
        # evidence_pool -> list of evidence strings
        # create claim embedding pair
        # pairs = [[f"[CLS] {claim} [SEP] {evidence}" for evidence in evidence_pool] for claim in x]
        pairs = []
        embeddings = []
        all_scores = []
        for i, claim in enumerate(x):
            evidences = evidence_pool[i]
            claim_pairs = [f"[CLS] {claim} [SEP] {evidence} [SEP]" for evidence in evidences]
            encoded = self.cross_encoder.encode(claim_pairs, convert_to_tensor=True)
            embeddings.append(encoded)
        
            # tokenized = self.tokenizer(claim_pairs, padding=True, truncation="longest_first", return_tensors="pt", max_length=100)
        # out = torch.softmax(self.mlp(embeddings), dim=1)
        return embeddings

# @hydra.main(config_path="../config", config_name="config")
# class RAV(pl.LightningModule):
#     def __init__(self, cfg: DictConfig):
#         super().__init__()
#         print("config", cfg)
#         self.retriever = Retriever(cfg)
#         self.ranker = Ranker(cfg)
        
@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    dataset = CSVDataset(file_path="/Users/rateria/Code/cs-5787-final-project/data/csv/generated_claim_triplets_with_topics.csv")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    retriever = Retriever(cfg)
    ranker = Ranker(cfg)
    attn = TransformerEncoder(num_layers=4, input_dim=768, num_heads=1, dim_feedforward=128)
    for batch in data_loader:
        print(batch)
        x, y = batch
        # rav = RAV()
        # print(rav)
        scores, indices, evidences = retriever(x)
        print(scores.shape, indices.shape, len(evidences))
        # print(f"\n{evidences}\n")
        out, h, s_hat = ranker(x, evidences)
        print(h.shape)
        enc_out = attn(h)
        print(enc_out.shape)
        break
        
if __name__ == "__main__":
    main()
    # test retriever
    # dataset = CSVDataset(file_path="data/csv/generated_claim_triplets_with_topics.csv")
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # retriever, ranker = main()
    
    # for batch in data_loader:
    #     print(batch)
    #     # rav = RAV()
    #     # print(rav)
    #     scores, indices, evidences = retriever(batch)
    #     print(scores, indices, evidences)
    #     break