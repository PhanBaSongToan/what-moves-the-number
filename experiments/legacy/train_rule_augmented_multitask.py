import csv
import json
import os
import random
import re
import unicodedata
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
OUTPUT_DIR = r"C:\Users\ACER\Downloads\vispam_outputs\rule_augmented_multitask_seed42"
MAX_LENGTH = 100
BATCH_SIZE = 16
EPOCHS = 10
ENCODER_LR = 1e-5
HEAD_LR = 5e-5
LAMBDA_TYPE = 1.0
PATIENCE = 3

ABBREVIATIONS = {
    "ko": "không", "k": "không", "kh": "không", "dc": "được",
    "đc": "được", "sp": "sản phẩm", "nv": "nhân viên", "shop": "cửa hàng",
    "mik": "mình", "m": "mình", "vs": "với", "r": "rồi", "ok": "tốt",
    "nhg": "nhưng", "ch": "chưa", "giao": "giao",
}
COIN_WORDS = ["xu", "nhận xu", "tích điểm", "voucher", "đánh giá nhận", "đổi quà"]


def seed_everything():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


def normalize_vi(text):
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    return " ".join(ABBREVIATIONS.get(tok, tok) for tok in tokens)


def rule_features(text):
    raw = str(text).lower().strip()
    norm = normalize_vi(raw)
    words = norm.split()
    repeated = bool(re.search(r"(.)\1{3,}", raw))
    emoji_or_symbols = len(re.findall(r"[^\w\sÀ-ỹ.,!?%]", raw, flags=re.UNICODE))
    return [
        min(len(words) / 50.0, 1.0),
        float(len(words) <= 3),
        float(bool(re.search(r"https?://|www\\.", raw))),
        float("@" in raw or "email" in raw),
        float(repeated),
        min(emoji_or_symbols / 10.0, 1.0),
        float(any(x in norm for x in COIN_WORDS)),
        float(len(set(words)) / max(len(words), 1)),
    ]


def load_official():
    with zipfile.ZipFile(ZIP_PATH) as z:
        return DatasetDict({
            "train": Dataset.from_pandas(pd.read_csv(z.open("train.csv")), preserve_index=False),
            "validation": Dataset.from_pandas(pd.read_csv(z.open("dev.csv")), preserve_index=False),
            "test": Dataset.from_pandas(pd.read_csv(z.open("test.csv")), preserve_index=False),
        })


class Reviews(torch.utils.data.Dataset):
    def __init__(self, split):
        self.ids = split["input_ids"]; self.mask = split["attention_mask"]
        self.rules = split["rule_features"]
        self.spam = [int(x != 0) for x in split["spam_label"]]
        self.spam_type = [max(int(x) - 1, 0) for x in split["spam_label"]]
        self.labels = split["spam_label"]

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.ids[i], dtype=torch.long),
            "attention_mask": torch.tensor(self.mask[i], dtype=torch.long),
            "rule_features": torch.tensor(self.rules[i], dtype=torch.float32),
            "binary_labels": torch.tensor(self.spam[i], dtype=torch.long),
            "type_labels": torch.tensor(self.spam_type[i], dtype=torch.long),
            "labels": torch.tensor(int(self.labels[i]), dtype=torch.long),
        }


class RuleAugmentedMultiTask(nn.Module):
    def __init__(self, rule_dim=8):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(CHECKPOINT)
        h = self.encoder.config.hidden_size
        self.rule_projection = nn.Sequential(nn.Linear(rule_dim, 64), nn.LayerNorm(64), nn.GELU())
        self.binary_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(h + 64, 2))
        self.type_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(h + 64, 3))

    def forward(self, input_ids, attention_mask, rule_features):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        fused = torch.cat([cls, self.rule_projection(rule_features)], dim=1)
        return self.binary_head(fused), self.type_head(fused)


def combined_predictions(binary_logits, type_logits):
    binary = binary_logits.argmax(1); spam_type = type_logits.argmax(1) + 1
    return torch.where(binary == 0, torch.zeros_like(spam_type), spam_type)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]; lb=lt=n=0.0
    for batch in loader:
        labels=batch["labels"].to(device); by=batch["binary_labels"].to(device); ty=batch["type_labels"].to(device)
        x={k:v.to(device) for k,v in batch.items() if k not in ["labels","binary_labels","type_labels"]}
        bl,tl=model(**x); mask=by==1
        lb += nn.functional.cross_entropy(bl,by).item()*len(labels)
        if mask.any():
            lt += nn.functional.cross_entropy(tl[mask],ty[mask]).item()*len(labels)
        pred=combined_predictions(bl,tl); ys.extend(labels.cpu().tolist()); ps.extend(pred.cpu().tolist()); n+=len(labels)
    ys=np.array(ys); ps=np.array(ps)
    return {"loss_binary":lb/n,"loss_type":lt/n,"accuracy":accuracy_score(ys,ps),"macro_f1":f1_score(ys,ps,average="macro",zero_division=0),"report":classification_report(ys,ps,zero_division=0)}


def main():
    seed_everything(); os.makedirs(OUTPUT_DIR,exist_ok=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    if device.type=="cuda": print("GPU:",torch.cuda.get_device_name(0))
    config={"method":"rule_augmented_multitask","seed":SEED,"checkpoint":CHECKPOINT,"dataset_zip":ZIP_PATH,"max_length":MAX_LENGTH,"batch_size":BATCH_SIZE,"epochs":EPOCHS,"encoder_lr":ENCODER_LR,"head_lr":HEAD_LR,"lambda_type":LAMBDA_TYPE,"patience":PATIENCE,"abbreviation_normalization":True,"rule_features":8}
    with open(os.path.join(OUTPUT_DIR,"config.json"),"w",encoding="utf-8") as f: json.dump(config,f,indent=2)
    raw=load_official(); tokenizer=AutoTokenizer.from_pretrained(CHECKPOINT,use_fast=False)
    def enrich(batch):
        normalized=[normalize_vi(x) for x in batch["comment"]]
        enc=tokenizer(normalized,truncation=True,padding="max_length",max_length=MAX_LENGTH)
        return {"input_ids":enc["input_ids"],"attention_mask":enc["attention_mask"],"rule_features":[rule_features(x) for x in batch["comment"]]}
    ds=raw.map(enrich,batched=True); keep=["input_ids","attention_mask","rule_features","spam_label"]
    ds=ds.remove_columns([c for c in ds["train"].column_names if c not in keep])
    train_loader=DataLoader(Reviews(ds["train"]),batch_size=BATCH_SIZE,shuffle=True); dev_loader=DataLoader(Reviews(ds["validation"]),batch_size=BATCH_SIZE); test_loader=DataLoader(Reviews(ds["test"]),batch_size=BATCH_SIZE)
    model=RuleAugmentedMultiTask().to(device); enc_params=list(model.encoder.parameters()); head_params=[p for n,p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer=torch.optim.AdamW([{"params":enc_params,"lr":ENCODER_LR},{"params":head_params,"lr":HEAD_LR}])
    log_path=os.path.join(OUTPUT_DIR,"training_log.csv")
    with open(log_path,"w",newline="",encoding="utf-8") as f: csv.writer(f).writerow(["epoch","train_loss","dev_loss_binary","dev_loss_type","dev_accuracy","dev_macro_f1"])
    best_f1=-1.; best_state=None; no_improve=0; best_epoch=0
    for epoch in range(1,EPOCHS+1):
        model.train(); running=steps=0
        for step,batch in enumerate(train_loader,1):
            by=batch.pop("binary_labels").to(device); ty=batch.pop("type_labels").to(device); batch.pop("labels"); x={k:v.to(device) for k,v in batch.items()}; optimizer.zero_grad(set_to_none=True)
            bl,tl=model(**x); mask=by==1
            loss_binary=nn.functional.cross_entropy(bl,by)
            loss_type=nn.functional.cross_entropy(tl[mask],ty[mask]) if mask.any() else torch.zeros((),device=device)
            loss=loss_binary+LAMBDA_TYPE*loss_type
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); running+=loss.item(); steps+=1
            if step%100==0: print(f"epoch={epoch} step={step} loss={running/steps:.4f}"); running=steps=0
        m=evaluate(model,dev_loader,device); print(f"epoch={epoch} dev_accuracy={m['accuracy']:.4f} dev_macro_f1={m['macro_f1']:.4f}")
        with open(os.path.join(OUTPUT_DIR,f"classification_report_epoch_{epoch}.txt"),"w",encoding="utf-8") as f: f.write(m["report"])
        with open(log_path,"a",newline="",encoding="utf-8") as f: csv.writer(f).writerow([epoch,"",m["loss_binary"],m["loss_type"],m["accuracy"],m["macro_f1"]])
        if m["macro_f1"]>best_f1: best_f1=m["macro_f1"]; best_epoch=epoch; no_improve=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; torch.save(best_state,os.path.join(OUTPUT_DIR,"best_model_state.pt"))
        else:
            no_improve+=1
            if no_improve>=PATIENCE: print(f"early_stopping epoch={epoch} best_epoch={best_epoch} best_dev_macro_f1={best_f1:.4f}"); break
    model.load_state_dict(best_state); test=evaluate(model,test_loader,device); result={"best_epoch":best_epoch,"best_dev_macro_f1":best_f1,"test_accuracy":test["accuracy"],"test_macro_f1":test["macro_f1"],"test_report":test["report"]}
    with open(os.path.join(OUTPUT_DIR,"final_test_metrics.json"),"w",encoding="utf-8") as f: json.dump(result,f,indent=2,ensure_ascii=False)
    with open(os.path.join(OUTPUT_DIR,"test_classification_report.txt"),"w",encoding="utf-8") as f: f.write(test["report"])
    print("\nRule-augmented multi-task test"); print(f"accuracy={test['accuracy']:.4f} macro_f1={test['macro_f1']:.4f}"); print(test["report"])


if __name__=="__main__": main()
