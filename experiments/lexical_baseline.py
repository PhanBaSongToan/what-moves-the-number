"""Lexical reference point: TF-IDF + linear SVM on the official split.

This is not part of the locked neural protocol and is not mixed into the neural
tables. It exists so the paper can state how much a bag-of-n-grams model gets on
the same split, which contextualises the PhoBERT numbers.
"""

from __future__ import annotations

import json
import os
import zipfile

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.environ.get("VISPAM_ZIP", r"C:\Users\ACER\Downloads\vispamdetectionv2_official.zip")

with zipfile.ZipFile(ZIP) as handle:
    import pandas as pd

    train = pd.read_csv(handle.open("train.csv"))
    test = pd.read_csv(handle.open("test.csv"))

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=200_000,
    sublinear_tf=True,
    lowercase=True,
)
X_train = vectorizer.fit_transform(train["comment"].astype(str))
X_test = vectorizer.transform(test["comment"].astype(str))

results = {"features": int(X_train.shape[1]), "vectorizer": "tfidf-1-2gram-min_df2-sublinear"}
for task, column, labels in (("task1", "label", 2), ("task2", "spam_label", 4)):
    model = LinearSVC(C=1.0)
    model.fit(X_train, train[column])
    prediction = model.predict(X_test)
    truth = test[column].to_numpy()
    results[task] = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "per_class_f1": {
            str(cls): float(
                f1_score(truth, prediction, labels=[cls], average="macro", zero_division=0)
            )
            for cls in range(labels)
        },
    }
    print(task, {k: round(v, 4) for k, v in results[task].items() if isinstance(v, float)})

with open(os.path.join(HERE, "lexical_baseline.json"), "w", encoding="utf-8") as handle:
    json.dump(results, handle, indent=2, ensure_ascii=False)
