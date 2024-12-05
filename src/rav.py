import os

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import remove_duplicate_strings

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

csv_file= "generated_claim_triplets_with_topics.csv"
csv_file_path = os.path.join("data", csv_file)
data = pd.read_csv(csv_file_path)

bi_encoder_model_name = "pritamdeka/PubMedBERT-mnli-snli-scinli-stsb"
cross_encoder_model_path = "/content/drive/MyDrive/fine_tuned_cross_encoder"

class Retriever(nn.Module):
    """Given a list of evidences and a claim, this returns the top-k evidences"""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.bi_encoder = SentenceTransformer(cfg.bi_encoder_model_name)
        self.k = cfg.k
        self.csv_file_path = os.path.join("data", cfg.csv_file)
        self.data = pd.read_csv(self.csv_file_path)
        self.evidence_pool = remove_duplicate_strings(data["Evidence"].dropna().tolist())
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
        cos_sim = torch.mm(x, self.evidence_embeddings)
        scores, indices = torch.topk(cos_sim, self.k, dim=1)
        evidences = [[self.evidence_pool[i] for i in row] for row in indices]
        return scores, indices, evidences
        
class Ranker(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cross_encoder = AutoModelForSequenceClassification.from_pretrained(cfg.cross_encoder_model_name)
        self.cross_encoder.classifier = nn.Identity()  # remove the last classifier layer
        self.mlp = nn.Sequential(
            nn.Linear(self.cross_encoder.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.cross_encoder_model_name)
        
    def forward(self, x, evidence_pool):
        # x -> b, claims
        # evidence_pool -> list of evidence strings
        # create claim embedding pair
        pairs = [[f"{claim} [SEP] {evidence}" for evidence in evidence_pool] for claim in x]
        tokenized = self.tokenizer(*pairs, padding=True, truncation="max_length", return_tensors="pt", max_length=100)
        h = self.cross_encoder(**tokenized, return_dict=False)
        s_hat = torch.softmax(self.mlp(h), dim=1)

@hydra.main(config_path="config", config_name="config")
class RAV(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()