import pandas as pd
import os
import requests
import logging
import math

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

COLUMNS = ["HITId","HITTypeId","Title","Description","Keywords","Reward","CreationTime","MaxAssignments","RequesterAnnotation","AssignmentDurationInSeconds","AutoApprovalDelayInSeconds","Expiration","NumberOfSimilarHITs","LifetimeInSeconds","AssignmentId","WorkerId","AssignmentStatus","AcceptTime","SubmitTime","AutoApprovalTime","ApprovalTime","RejectionTime","RequesterFeedback","WorkTimeInSeconds","LifetimeApprovalRate","Last30DaysApprovalRate","Last7DaysApprovalRate","Input.corpus_id","Input.file_id","Input.token_id","Input.sentence","Input.token","Input.begin","Input.end","Answer.sentiment.label","Approve","Reject"]


def load(path):
    raw = pd.read_csv(path, names=COLUMNS, low_memory=False)
    logging.info(f"Loaded {len(raw):,} rows from {path}")
    return raw


def select_columns(raw):
    df = raw[list(KEEP.keys())].rename(columns=KEEP).copy()
    return df


def map_labels(df):
    before = len(df)
    df["complexity"] = df["complexity"].map(LABEL_MAP)
    unmapped = df["complexity"].isna().sum()
    if unmapped:
        logging.warning(f"{unmapped:,} rows had unmapped labels and will be dropped")
    df = df.dropna().reset_index(drop=True)
    df["complexity"] = df["complexity"].astype(int)
    logging.info(f"Rows after label mapping: {len(df):,} ({before - len(df):,} dropped)")
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

def preprocess_data(dataset: DatasetDict):
    """
    Filter out rows with missing values or invalid complexity labels.

    Params:
        dataset: DatasetDict with 'train' and 'test' splits

    Returns:
        Filtered DatasetDict with only valid rows
    """
    def no_missing(row):
        for v in row.values():
            if v is None:
                return False
            if isinstance(v, float) and math.isnan(v):
                return False
            if isinstance(v, str) and v.strip() == "":
                return False
        return True

    # filter out rows that contain any empty value
    train = dataset["train"].filter(
        lambda row: no_missing(row) 
    )
   # filter out rows that contain any empty value
    test = dataset["test"].filter(
        lambda row: no_missing(row) 
    )
    return DatasetDict(dict(train=train, test=test)) 

