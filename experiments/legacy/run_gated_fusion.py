import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

SEED = 42
CHECKPOINT = "vinai/phobert-base-v2"
BATCH_SIZE = 32
MAX_LENGTH = 128
TAU = 0.15


def seed_everything():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


class Reviews(Dataset):
    def __init__(self, split, cat_to_id):
        self.comment_ids = split["comment_ids"]
        self.comment_mask = split["comment_mask"]
        self.desc_ids = split["desc_ids"]
        self.desc_mask = split["desc_mask"]
        self.category = [cat_to_id.get(x, 0) for x in split["category"]]
        self.labels = split["spam_label"]

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return {
            "comment_ids": torch.tensor(self.comment_ids[i], dtype=torch.long),
            "comment_mask": torch.tensor(self.comment_mask[i], dtype=torch.long),
            "desc_ids": torch.tensor(self.desc_ids[i], dtype=torch.long),
            "desc_mask": torch.tensor(self.desc_mask[i], dtype=torch.long),
            "category": torch.tensor(self.category[i], dtype=torch.long),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }


def encode(batch, tokenizer, text_key):
    out = tokenizer(batch[text_key], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    return {f"{text_key}_ids": out["input_ids"], f"{text_key}_mask": out["attention_mask"]}


def mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)


@torch.no_grad()
def precompute_description(model, loader, device):
    model.eval(); result = []
    for batch in loader:
        ids = batch["desc_ids"].to(device); mask = batch["desc_mask"].to(device)
        result.append(mean_pool(model(input_ids=ids, attention_mask=mask).last_hidden_state, mask).cpu())
    return torch.cat(result)


class GatedClassifier(nn.Module):
    def __init__(self, base, desc_features, num_categories):
        super().__init__(); self.base = base; self.desc_features = desc_features
        h = base.config.hidden_size
        self.cat = nn.Embedding(num_categories, 32)
        self.gate = nn.Sequential(nn.Linear(h * 2 + 33, h), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(h + 33, 4))

    def forward(self, batch, indices):
        cm = self.base(input_ids=batch["comment_ids"], attention_mask=batch["comment_mask"])
        c = mean_pool(cm.last_hidden_state, batch["comment_mask"])
        d = self.desc_features[indices.detach().cpu()].to(c.device)
        sim = F.cosine_similarity(c, d, dim=1).unsqueeze(1)
        cat = self.cat(batch["category"])
        g = self.gate(torch.cat([c, d, sim, cat], dim=1))
        fused = c + g * d
        return self.classifier(torch.cat([fused, sim, cat], dim=1))


def evaluate(model, loader, device, adjustment):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for indices, batch in loader:
            labels=batch["labels"].numpy(); x={k:v.to(device) for k,v in batch.items() if k!="labels"}
            logits=model(x, indices.to(device)) + adjustment
            ps.extend(logits.argmax(1).cpu().tolist()); ys.extend(labels.tolist())
    return np.array(ys), np.array(ps)


def main():
    seed_everything(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    raw=load_dataset("sonlam1102/vispamdetection2")
    tokenizer=AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=False)
    cats=sorted(set(raw["train"]["category"])); cat_to_id={c:i for i,c in enumerate(cats)}

    def tok(batch):
        a=tokenizer(batch["comment"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
        b=tokenizer(batch["description"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
        return {"comment_ids":a["input_ids"],"comment_mask":a["attention_mask"],"desc_ids":b["input_ids"],"desc_mask":b["attention_mask"]}
    ds=raw.map(tok, batched=True)
    keep=["comment_ids","comment_mask","desc_ids","desc_mask","category","spam_label"]
    ds=ds.remove_columns([c for c in ds["train"].column_names if c not in keep])

    # Precompute description representations once using a frozen PhoBERT.
    encoder=AutoModel.from_pretrained(CHECKPOINT).to(device)
    temp_loaders={n:DataLoader(Reviews(ds[n],cat_to_id),batch_size=BATCH_SIZE) for n in ["train","validation","test"]}
    desc_features={n:precompute_description(encoder,temp_loaders[n],device) for n in ["train","validation","test"]}
    del encoder; torch.cuda.empty_cache()

    # Rebuild datasets/loaders with stable indices into precomputed description vectors.
    def loader(name, shuffle=False):
        base=Reviews(ds[name],cat_to_id); feats=desc_features[name]
        class Indexed(Dataset):
            def __len__(self): return len(base)
            def __getitem__(self,i): return i, base[i]
        return DataLoader(Indexed(),batch_size=BATCH_SIZE,shuffle=shuffle), feats
    train_loader,train_desc=loader("train",True); dev_loader,dev_desc=loader("validation"); test_loader,test_desc=loader("test")
    model=GatedClassifier(AutoModel.from_pretrained(CHECKPOINT),train_desc,len(cats)).to(device)
    counts=np.bincount(np.asarray(raw["train"]["spam_label"]),minlength=4).astype(float); priors=counts/counts.sum()
    adjustment=torch.tensor((-TAU*np.log(priors)),dtype=torch.float32,device=device)
    adjustment = adjustment - adjustment[0]
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5); best=None; best_f1=-1
    for epoch in range(1,3):
        model.desc_features = train_desc
        model.train(); running=0
        for step,(indices,batch) in enumerate(train_loader,1):
            y=batch.pop("labels").to(device); x={k:v.to(device) for k,v in batch.items()}
            opt.zero_grad(set_to_none=True); logits=model(x,indices.to(device)); loss=F.cross_entropy(logits+adjustment,y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); running+=loss.item()
            if step%100==0: print(f"epoch={epoch} step={step} loss={running/100:.4f}"); running=0
        model.desc_features = dev_desc
        y,p=evaluate(model,dev_loader,device,adjustment); f=f1_score(y,p,average="macro",zero_division=0)
        print(f"epoch={epoch} dev_accuracy={accuracy_score(y,p):.4f} dev_macro_f1={f:.4f}")
        if f>best_f1: best_f1=f; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best)
    model.desc_features = test_desc
    y,p=evaluate(model,test_loader,device,adjustment)
    print("\nGated fusion + logit adjustment")
    print("accuracy=%.4f macro_f1=%.4f"%(accuracy_score(y,p),f1_score(y,p,average="macro",zero_division=0)))
    print(classification_report(y,p,zero_division=0))


if __name__=="__main__": main()
