import os
import random
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

SEED = 42
CHECKPOINT = "vinai/phobert-base-v2"
ZIP_PATH = r"C:\Users\ACER\Downloads\vispamdetectionv2_official.zip"
OUTPUT_DIR = r"C:\Users\ACER\Downloads\vispam_outputs\attentive_pooling_task2_seed42"


def seed_everything():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


def load_official():
    with zipfile.ZipFile(ZIP_PATH) as z:
        return DatasetDict({
            "train": Dataset.from_pandas(pd.read_csv(z.open("train.csv")), preserve_index=False),
            "validation": Dataset.from_pandas(pd.read_csv(z.open("dev.csv")), preserve_index=False),
            "test": Dataset.from_pandas(pd.read_csv(z.open("test.csv")), preserve_index=False),
        })


class TorchReviews(torch.utils.data.Dataset):
    def __init__(self, split):
        self.ids = split["input_ids"]; self.mask = split["attention_mask"]; self.labels = split["spam_label"]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {"input_ids": torch.tensor(self.ids[i], dtype=torch.long),
                "attention_mask": torch.tensor(self.mask[i], dtype=torch.long),
                "labels": torch.tensor(self.labels[i], dtype=torch.long)}


class AttentivePhoBERT(nn.Module):
    def __init__(self, num_labels=4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(CHECKPOINT)
        h = self.encoder.config.hidden_size
        self.attention = nn.Sequential(nn.Linear(h, 128), nn.Tanh(), nn.Linear(128, 1))
        self.gate = nn.Sequential(nn.Linear(h * 3, h), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.LayerNorm(h * 3), nn.Dropout(0.2), nn.Linear(h * 3, h),
            nn.GELU(), nn.Dropout(0.2), nn.Linear(h, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.bool().unsqueeze(-1)
        mean = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        max_hidden = hidden.masked_fill(~mask, -1e4).max(1).values
        scores = self.attention(hidden).masked_fill(~mask, -1e4).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        attn = (hidden * weights).sum(1)
        pooled = torch.cat([mean, max_hidden, attn], dim=1)
        pooled = pooled * (0.5 + 0.5 * self.gate(pooled).repeat(1, 3))
        return self.classifier(pooled)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]
    for batch in loader:
        y=batch.pop("labels"); x={k:v.to(device) for k,v in batch.items()}
        ps.extend(model(**x).argmax(1).cpu().tolist()); ys.extend(y.tolist())
    return np.array(ys), np.array(ps)


def main():
    seed_everything(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda": print("GPU:", torch.cuda.get_device_name(0))
    raw=load_official(); tokenizer=AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=False)
    def tok(batch): return tokenizer(batch["comment"], truncation=True, padding="max_length", max_length=100)
    ds=raw.map(tok, batched=True)
    keep=["input_ids","attention_mask","spam_label"]
    ds=ds.remove_columns([c for c in ds["train"].column_names if c not in keep])
    train_loader=DataLoader(TorchReviews(ds["train"]), batch_size=16, shuffle=True)
    dev_loader=DataLoader(TorchReviews(ds["validation"]), batch_size=16)
    test_loader=DataLoader(TorchReviews(ds["test"]), batch_size=16)
    model=AttentivePhoBERT().to(device)
    loss_fn=nn.CrossEntropyLoss()
    opt=torch.optim.AdamW(model.parameters(), lr=2e-5)
    best_f1=-1; best_state=None; no_improve=0
    for epoch in range(1,11):
        model.train(); running=0
        for step,batch in enumerate(train_loader,1):
            y=batch.pop("labels").to(device); x={k:v.to(device) for k,v in batch.items()}
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(**x),y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); running+=loss.item()
            if step%100==0: print(f"epoch={epoch} step={step} loss={running/100:.4f}"); running=0
        y,p=evaluate(model,dev_loader,device); f=f1_score(y,p,average="macro",zero_division=0)
        print(f"epoch={epoch} dev_accuracy={accuracy_score(y,p):.4f} dev_macro_f1={f:.4f}")
        if f>best_f1:
            best_f1=f; no_improve=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            no_improve+=1
            if no_improve>=3: print(f"early_stopping best_dev_macro_f1={best_f1:.4f}"); break
    model.load_state_dict(best_state); y,p=evaluate(model,test_loader,device)
    print("\nAttentive pooling test")
    print("accuracy=%.4f macro_f1=%.4f"%(accuracy_score(y,p),f1_score(y,p,average="macro",zero_division=0)))
    print(classification_report(y,p,zero_division=0))
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    torch.save(model.state_dict(),os.path.join(OUTPUT_DIR,"model.pt"))


if __name__=="__main__": main()
