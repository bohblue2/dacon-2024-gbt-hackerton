import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple

def load_data(root_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(f"{root_path}/train.csv")
    test_df = pd.read_csv(f"{root_path}/test.csv")
    return train_df, test_df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['제목_키워드'] = df['키워드'].apply(lambda x: ' '.join(list(dict.fromkeys(x.split(",")))))
    return df

def get_label_encoded(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    label_encoder = {label: i for i, label in enumerate(df['분류'].unique())}
    df['label'] = df['분류'].map(label_encoder)
    return df, label_encoder

def split_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_size, stratify=df['분류'], random_state=42)