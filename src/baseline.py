import argparse
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from collections import Counter
import numpy as np
import json
import os

def majority_baseline(train_texts, train_labels, test_labels):
    majority_label = Counter(train_labels).most_common(1)[0][0]
    preds = [majority_label] * len(test_labels)
    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1, "majority_label": int(majority_label)}

def tfidf_logreg_baseline(train_texts, train_labels, test_texts, test_labels):
    vect = TfidfVectorizer(min_df=2, ngram_range=(1,2), max_features=30000)
    X_train = vect.fit_transform(train_texts)
    X_test = vect.transform(test_texts)
    clf = LogisticRegression(max_iter=2000, n_jobs=None, verbose=0)
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)
    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()

    ds = load_dataset("banking77")
    train = ds["train"]
    test = ds["test"]

    train_texts = train["text"]
    train_labels = train["label"]
    test_texts = test["text"]
    test_labels = test["label"]

    os.makedirs(args.reports_dir, exist_ok=True)

    maj = majority_baseline(train_texts, train_labels, test_labels)
    with open(os.path.join(args.reports_dir, "baseline_majority.json"), "w") as f:
        json.dump(maj, f, indent=2)
    print("Majority baseline:", maj)

    tfidf = tfidf_logreg_baseline(train_texts, train_labels, test_texts, test_labels)
    with open(os.path.join(args.reports_dir, "baseline_tfidf_logreg.json"), "w") as f:
        json.dump(tfidf, f, indent=2)
    print("TF-IDF + Logistic Regression:", tfidf)

if __name__ == "__main__":
    main()
