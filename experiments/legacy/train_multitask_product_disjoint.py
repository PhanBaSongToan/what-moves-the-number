import csv, json, os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModel, AutoTokenizer

SEED=42
CHECKPOINT="vinai/phobert-base-v2"
DATA_DIR=r"C:\Users\ACER\Downloads\vispam_outputs\product_disjoint_split_seed42"
OUT=r"C:\Users\ACER\Downloads\vispam_outputs\multitask_product_disjoint_seed42"
MAX_LENGTH=100; BATCH_SIZE=16; EPOCHS=10; ENC_LR=1e-5; HEAD_LR=5e-5; PATIENCE=3

def seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def load_data():
    return DatasetDict({n:Dataset.from_pandas(pd.read_csv(os.path.join(DATA_DIR,n+'.csv')),preserve_index=False) for n in ['train','dev','test']})

class Reviews(TorchDataset):
    def __init__(self,s):
        self.ids=s['input_ids']; self.mask=s['attention_mask']; self.labels=s['spam_label']
    def __len__(self): return len(self.labels)
    def __getitem__(self,i):
        y=int(self.labels[i]); return {'input_ids':torch.tensor(self.ids[i],dtype=torch.long),'attention_mask':torch.tensor(self.mask[i],dtype=torch.long),'binary_labels':torch.tensor(int(y!=0),dtype=torch.long),'type_labels':torch.tensor(max(y-1,0),dtype=torch.long),'labels':torch.tensor(y,dtype=torch.long)}

class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=AutoModel.from_pretrained(CHECKPOINT); h=self.encoder.config.hidden_size
        self.binary_head=nn.Sequential(nn.Dropout(.2),nn.Linear(h,2)); self.type_head=nn.Sequential(nn.Dropout(.2),nn.Linear(h,3))
    def forward(self,input_ids,attention_mask):
        cls=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:,0]
        return self.binary_head(cls),self.type_head(cls)

def combine(bl,tl):
    b=bl.argmax(1); t=tl.argmax(1)+1; return torch.where(b==0,torch.zeros_like(t),t)

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); ys=[]; ps=[]; lb=lt=n=0.
    for batch in loader:
        y=batch['labels'].to(device); by=batch['binary_labels'].to(device); ty=batch['type_labels'].to(device); x={k:v.to(device) for k,v in batch.items() if k not in ['labels','binary_labels','type_labels']}
        bl,tl=model(**x); mask=by==1; lb+=nn.functional.cross_entropy(bl,by).item()*len(y)
        if mask.any(): lt+=nn.functional.cross_entropy(tl[mask],ty[mask]).item()*len(y)
        p=combine(bl,tl); ys+=y.cpu().tolist(); ps+=p.cpu().tolist(); n+=len(y)
    return {'loss_binary':lb/n,'loss_type':lt/n,'accuracy':accuracy_score(ys,ps),'macro_f1':f1_score(ys,ps,average='macro',zero_division=0),'report':classification_report(ys,ps,zero_division=0)}

def main():
    seed(); os.makedirs(OUT,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',device)
    raw=load_data(); tok=AutoTokenizer.from_pretrained(CHECKPOINT,use_fast=False)
    def encode(b): return tok(b['comment'],truncation=True,padding='max_length',max_length=MAX_LENGTH)
    ds=raw.map(encode,batched=True); keep=['input_ids','attention_mask','spam_label']; ds=ds.remove_columns([c for c in ds['train'].column_names if c not in keep])
    tr=DataLoader(Reviews(ds['train']),batch_size=BATCH_SIZE,shuffle=True); dv=DataLoader(Reviews(ds['dev']),batch_size=BATCH_SIZE); te=DataLoader(Reviews(ds['test']),batch_size=BATCH_SIZE)
    model=Model().to(device); opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':ENC_LR},{'params':list(model.binary_head.parameters())+list(model.type_head.parameters()),'lr':HEAD_LR}]); ce=nn.CrossEntropyLoss(); log=os.path.join(OUT,'training_log.csv')
    cfg={'seed':SEED,'checkpoint':CHECKPOINT,'data_dir':DATA_DIR,'max_length':MAX_LENGTH,'batch_size':BATCH_SIZE,'epochs':EPOCHS,'encoder_lr':ENC_LR,'head_lr':HEAD_LR,'patience':PATIENCE}
    with open(os.path.join(OUT,'config.json'),'w') as f: json.dump(cfg,f,indent=2)
    with open(log,'w',newline='') as f: csv.writer(f).writerow(['epoch','train_loss','dev_loss_binary','dev_loss_type','dev_accuracy','dev_macro_f1'])
    best=-1; state=None; wait=0; best_epoch=0
    for ep in range(1,EPOCHS+1):
        model.train(); total=steps=0.
        for step,b in enumerate(tr,1):
            by=b.pop('binary_labels').to(device); ty=b.pop('type_labels').to(device); b.pop('labels'); x={k:v.to(device) for k,v in b.items()}; opt.zero_grad(set_to_none=True); bl,tl=model(**x); mask=by==1; l=ce(bl,by)+(ce(tl[mask],ty[mask]) if mask.any() else torch.zeros((),device=device)); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=l.item(); steps+=1
        m=evaluate(model,dv,device); print(f'epoch={ep} train_loss={total/steps:.4f} dev_accuracy={m["accuracy"]:.4f} dev_macro_f1={m["macro_f1"]:.4f}')
        with open(os.path.join(OUT,f'classification_report_epoch_{ep}.txt'),'w',encoding='utf-8') as f: f.write(m['report'])
        with open(log,'a',newline='') as f: csv.writer(f).writerow([ep,total/steps,m['loss_binary'],m['loss_type'],m['accuracy'],m['macro_f1']])
        if m['macro_f1']>best: best=m['macro_f1']; best_epoch=ep; wait=0; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; torch.save(state,os.path.join(OUT,'best_model_state.pt'))
        else:
            wait+=1
            if wait>=PATIENCE: print('early_stopping'); break
    model.load_state_dict(state); test=evaluate(model,te,device); result={'best_epoch':best_epoch,'best_dev_macro_f1':best,'test_accuracy':test['accuracy'],'test_macro_f1':test['macro_f1'],'test_report':test['report']}
    with open(os.path.join(OUT,'final_test_metrics.json'),'w',encoding='utf-8') as f: json.dump(result,f,indent=2,ensure_ascii=False)
    with open(os.path.join(OUT,'test_classification_report.txt'),'w',encoding='utf-8') as f: f.write(test['report'])
    print('\nProduct-disjoint test'); print(f'accuracy={test["accuracy"]:.4f} macro_f1={test["macro_f1"]:.4f}'); print(test['report'])

if __name__=='__main__': main()
