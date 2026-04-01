import pandas as pd
import os
import requests
import logging

from datasets import Dataset, DatasetDict

KEEP = {
    "HITId": "task_id",
    "WorkerId": "annotator_id",
    "Input.corpus_id": "corpus",
    "Input.sentence": "sentence",
    "Input.token": "token",
    "Answer.sentiment.label": "complexity",
}

LABEL_MAP = {
    "Very Easy": 0,
    "Easy": 1,
    "Neutral": 2,
    "Difficult": 3,
    "Very Difficult": 4,
}


def load(path):
    raw = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(raw):,} rows from {path}")
    return raw


def select_columns(raw):
    df = raw[list(KEEP.keys())].rename(columns=KEEP).copy()
    print(f"Selected {len(df.columns)} columns: {list(df.columns)}")
    return df


def map_labels(df):
    before = len(df)
    df["complexity"] = df["complexity"].map(LABEL_MAP)
    unmapped = df["complexity"].isna().sum()
    if unmapped:
        print(f"Warning: {unmapped:,} rows had unmapped labels and will be dropped")
    df = df.dropna().reset_index(drop=True)
    df["complexity"] = df["complexity"].astype(int)
    print(f"Rows after label mapping: {len(df):,} ({before - len(df):,} dropped)")
    return df


def load_dataset(cache_dir: str = "./data/per_annotator", test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    """
    Load the per-annotator lexical complexity dataset as a Hugging Face DatasetDict.

    Downloads from https://github.com/MMU-TDMLab/LCP_Subjectivity if not already cached.

    Params:
        cache_dir: Directory to save/load the dataset file
        test_size: Fraction of data to use as test set
        seed: Random seed for reproducibility

    Returns:
        DatasetDict with keys 'train' and 'test'
    """
    url = "https://raw.githubusercontent.com/MMU-TDMLab/LCP_Subjectivity/master/LCP_2021/batchResults/all.csv"
    local_path = os.path.join(cache_dir, "all.csv")

    os.makedirs(cache_dir, exist_ok=True)

    if not os.path.exists(local_path):
        logging.info("Downloading per-annotator data")
        response = requests.get(url)
        if not response.ok:
            raise RuntimeError(f"Failed to download per-annotator data: {response.status_code}")
        with open(local_path, "w") as f:
            f.write(response.text)

    raw = load(local_path)
    df = select_columns(raw)
    df = map_labels(df)

    train_df = df.sample(frac=1 - test_size, random_state=seed)
    test_df = df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    return DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False),
    })


def save(df, path):
    df.to_csv(path, index=False)
    check = pd.read_csv(path)
    assert check.shape == df.shape
    print(f"\nSaved to {path} — {df.shape[0]:,} rows x {df.shape[1]} columns")


