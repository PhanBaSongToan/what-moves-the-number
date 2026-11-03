import zipfile

import pandas as pd

ZIP = r"C:\Users\ACER\Downloads\vispamdetectionv2_official.zip"
DISJOINT = r"C:\Users\ACER\Downloads\vispam_outputs\product_disjoint_split_seed42"

with zipfile.ZipFile(ZIP) as z:
    parts = {name: pd.read_csv(z.open(name + ".csv")) for name in ["train", "dev", "test"]}

print("=== official split ===")
for name, df in parts.items():
    print(name, len(df), df["spam_label"].value_counts().sort_index().to_dict())

print("\n=== product-disjoint split ===")
for name in ["train", "dev", "test"]:
    df = pd.read_csv(rf"{DISJOINT}\{name}.csv")
    print(name, len(df), df["spam_label"].value_counts().sort_index().to_dict())

train_c = parts["train"]["comment"].astype(str).str.strip().str.lower()
test_c = parts["test"]["comment"].astype(str).str.strip().str.lower()
print("\nexact duplicate comments test^train:", len(set(test_c) & set(train_c)))
print("test rows whose comment appears in train:", int(test_c.isin(set(train_c)).sum()))

all_c = pd.concat([p["comment"].astype(str).str.strip().str.lower() for p in parts.values()])
print("total rows:", len(all_c), "unique comments:", all_c.nunique())
print("rows in comment groups of size>1:", int(all_c.duplicated(keep=False).sum()))

# product-disjoint guarantee check
def norm(x):
    import re
    import unicodedata

    x = unicodedata.normalize("NFKC", str(x)).lower().strip()
    return re.sub(r"\s+", " ", x)


tr = pd.read_csv(rf"{DISJOINT}\train.csv")
te = pd.read_csv(rf"{DISJOINT}\test.csv")
dv = pd.read_csv(rf"{DISJOINT}\dev.csv")
key = lambda d: set(d["product_name"].map(norm) + "||" + d["category"].map(norm) + "||" + d["description"].map(norm))
print("disjoint group overlap train/test:", len(key(tr) & key(te)), "train/dev:", len(key(tr) & key(dv)), "dev/test:", len(key(dv) & key(te)))

# comment overlap in disjoint split
dtr = tr["comment"].astype(str).str.strip().str.lower()
dte = te["comment"].astype(str).str.strip().str.lower()
print("disjoint test rows whose comment appears in disjoint train:", int(dte.isin(set(dtr)).sum()))

# class-1 composition
print("\nclass1 in official train:", int((parts['train']['spam_label']==1).sum()), "unique comments:", parts['train'].loc[parts['train']['spam_label']==1,'comment'].nunique())
