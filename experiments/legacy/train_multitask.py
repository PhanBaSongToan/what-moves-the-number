import csv
import json
import os
import random
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModel, AutoTokenizer

SEED = 42
CHECKPOINT = "vinai/phobert-base-v2"
ZIP_PATH = r"C:\Users\ACER\Downloads\vispamdetectionv2_official.zip"
OUTPUT_DIR = r"C:\Users\ACER\Downloads\vispam_outputs\multitask_type_weighted_task2_seed42"
MAX_LENGTH = 100
BATCH_SIZE = 16
EPOCHS = 10
ENCODER_LR = 1e-5
HEAD_LR = 5e-5
LAMBDA_TYPE = 1.0
PATIENCE = 3


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


class Reviews(TorchDataset):
    def __init__(self, split):
        self.ids = split["input_ids"]; self.mask = split["attention_mask"]
        self.spam = [int(x != 0) for x in split["spam_label"]]
        self.spam_type = [max(int(x) - 1, 0) for x in split["spam_label"]]
        self.labels = split["spam_label"]

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.ids[i], dtype=torch.long),
            "attention_mask": torch.tensor(self.mask[i], dtype=torch.long),
            "binary_labels": torch.tensor(self.spam[i], dtype=torch.long),
            "type_labels": torch.tensor(self.spam_type[i], dtype=torch.long),
            "labels": torch.tensor(int(self.labels[i]), dtype=torch.long),
        }


class MultiTaskPhoBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(CHECKPOINT)
        h = self.encoder.config.hidden_size
        self.binary_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(h, 2))
        self.type_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(h, 3))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.binary_head(cls), self.type_head(cls)


def combined_predictions(binary_logits, type_logits):
    binary = binary_logits.argmax(1)
    spam_type = type_logits.argmax(1) + 1
    return torch.where(binary == 0, torch.zeros_like(spam_type), spam_type)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]; total_binary=0.; total_type=0.; n=0
    for batch in loader:
        labels=batch["labels"].to(device)
        binary_y=batch["binary_labels"].to(device)
        type_y=batch["type_labels"].to(device)
        x={k:v.to(device) for k,v in batch.items() if k not in ["labels","binary_labels","type_labels"]}
        binary_logits,type_logits=model(**x)
        binary_loss=nn.functional.cross_entropy(binary_logits,binary_y)
        mask=binary_y==1
        type_loss=nn.functional.cross_entropy(type_logits[mask],type_y[mask]) if mask.any() else torch.tensor(0.,device=device)
        pred=combined_predictions(binary_logits,type_logits)
        ys.extend(labels.cpu().tolist()); ps.extend(pred.cpu().tolist())
        total_binary+=binary_loss.item()*len(labels); total_type+=type_loss.item()*len(labels); n+=len(labels)
    ys=np.array(ys); ps=np.array(ps)
    return {
        "loss_binary": total_binary/n,
        "loss_type": total_type/n,
        "accuracy": accuracy_score(ys,ps),
        "macro_f1": f1_score(ys,ps,average="macro",zero_division=0),
        "report": classification_report(ys,ps,zero_division=0),
    }


def main():
    seed_everything(); os.makedirs(OUTPUT_DIR, exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    if device.type=="cuda": print("GPU:",torch.cuda.get_device_name(0))
    config={"method":"multitask_type_weighted","seed":SEED,"checkpoint":CHECKPOINT,"dataset_zip":ZIP_PATH,"max_length":MAX_LENGTH,"batch_size":BATCH_SIZE,"epochs":EPOCHS,"encoder_lr":ENCODER_LR,"head_lr":HEAD_LR,"lambda_type":LAMBDA_TYPE,"patience":PATIENCE}
    with open(os.path.join(OUTPUT_DIR,"config.json"),"w",encoding="utf-8") as f: json.dump(config,f,indent=2)
    raw=load_official(); tokenizer=AutoTokenizer.from_pretrained(CHECKPOINT,use_fast=False)
    def tok(batch): return tokenizer(batch["comment"],truncation=True,padding="max_length",max_length=MAX_LENGTH)
    ds=raw.map(tok,batched=True); keep=["input_ids","attention_mask","spam_label"]
    ds=ds.remove_columns([c for c in ds["train"].column_names if c not in keep])
    train_loader=DataLoader(Reviews(ds["train"]),batch_size=BATCH_SIZE,shuffle=True)
    dev_loader=DataLoader(Reviews(ds["validation"]),batch_size=BATCH_SIZE)
    test_loader=DataLoader(Reviews(ds["test"]),batch_size=BATCH_SIZE)
    model=MultiTaskPhoBERT().to(device)
    encoder_params=list(model.encoder.parameters()); head_params=list(model.binary_head.parameters())+list(model.type_head.parameters())
    optimizer=torch.optim.AdamW([{"params":encoder_params,"lr":ENCODER_LR},{"params":head_params,"lr":HEAD_LR}])
    binary_loss=nn.CrossEntropyLoss()
    spam_train_labels=[int(x)-1 for x in raw["train"]["spam_label"] if int(x)!=0]
    spam_counts=np.bincount(spam_train_labels,minlength=3).astype(float)
    type_weights=len(spam_train_labels)/(3*np.maximum(spam_counts,1))
    type_weights=torch.tensor(type_weights,dtype=torch.float32,device=device)
    type_loss=nn.CrossEntropyLoss(weight=type_weights)
    log_path=os.path.join(OUTPUT_DIR,"training_log.csv")
    with open(log_path,"w",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch","train_loss","dev_loss_binary","dev_loss_type","dev_accuracy","dev_macro_f1"])
    best_f1=-1.; best_state=None; no_improve=0; best_epoch=0
    for epoch in range(1,EPOCHS+1):
        model.train(); running=0.; steps=0; epoch_loss=0.; epoch_steps=0
        for step,batch in enumerate(train_loader,1):
            binary_y=batch.pop("binary_labels").to(device); type_y=batch.pop("type_labels").to(device); batch.pop("labels")
            x={k:v.to(device) for k,v in batch.items()}; optimizer.zero_grad(set_to_none=True)
            binary_logits,type_logits=model(**x); mask=binary_y==1
            loss_b=binary_loss(binary_logits,binary_y)
            loss_t=type_loss(type_logits[mask],type_y[mask]) if mask.any() else torch.zeros((),device=device)
            loss=loss_b+LAMBDA_TYPE*loss_t; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            running+=loss.item(); steps+=1; epoch_loss+=loss.item(); epoch_steps+=1
            if step%100==0: print(f"epoch={epoch} step={step} loss={running/steps:.4f}"); running=0.; steps=0
        metrics=evaluate(model,dev_loader,device)
        print(f"epoch={epoch} dev_accuracy={metrics['accuracy']:.4f} dev_macro_f1={metrics['macro_f1']:.4f}")
        with open(os.path.join(OUTPUT_DIR,f"classification_report_epoch_{epoch}.txt"),"w",encoding="utf-8") as f: f.write(metrics["report"])
        with open(log_path,"a",newline="",encoding="utf-8") as f: csv.writer(f).writerow([epoch,epoch_loss/max(epoch_steps,1),metrics["loss_binary"],metrics["loss_type"],metrics["accuracy"],metrics["macro_f1"]])
        if metrics["macro_f1"]>best_f1:
            best_f1=metrics["macro_f1"]; best_epoch=epoch; no_improve=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(best_state,os.path.join(OUTPUT_DIR,"best_model_state.pt"))
        else:
            no_improve+=1
            if no_improve>=PATIENCE: print(f"early_stopping epoch={epoch} best_epoch={best_epoch} best_dev_macro_f1={best_f1:.4f}"); break
    model.load_state_dict(best_state); test=evaluate(model,test_loader,device)
    result={"best_epoch":best_epoch,"best_dev_macro_f1":best_f1,"test_accuracy":test["accuracy"],"test_macro_f1":test["macro_f1"],"test_report":test["report"]}
    with open(os.path.join(OUTPUT_DIR,"final_test_metrics.json"),"w",encoding="utf-8") as f: json.dump(result,f,indent=2,ensure_ascii=False)
    with open(os.path.join(OUTPUT_DIR,"test_classification_report.txt"),"w",encoding="utf-8") as f: f.write(test["report"])
    print("\nMulti-task test"); print(f"accuracy={test['accuracy']:.4f} macro_f1={test['macro_f1']:.4f}"); print(test["report"])


if __name__=="__main__": main()
