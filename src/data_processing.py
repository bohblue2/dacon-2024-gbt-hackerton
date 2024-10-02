import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple

def load_data(root_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(f"{root_path}/train.csv")
    test_df = pd.read_csv(f"{root_path}/test.csv")
    return train_df, test_df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    import re
    def normalize_title(text):
        text = re.sub(r'\s+', ' ', text).strip()
        return text.strip()

    def normalize_keywords(text):
        text = re.sub(r'[^가-힣\s,]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.strip()

    def clean_keywords(keywords):
        keywords = keywords.split(',')
        keywords = [keyword.strip() for keyword in keywords if keyword.strip()]
        return ' '.join(keywords)

    df['title'] = df['제목'].apply(normalize_title)
    df['keywords'] = df['키워드'].apply(lambda x: clean_keywords(normalize_keywords(x)))
    df['text'] = df['title'] + ' [SEP] ' + df['keywords']
    return df

def get_label_encoded(df: pd.DataFrame, model_type: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    # secondary_df = train_df[train_df['분류'] != '지역'].copy()
    # primary_df = train_df.copy()
    # primary_df.loc[primary_df['분류'] != '지역', '분류'] = '비지역'
    if model_type == "primary":
        primary_df = df.copy()
        primary_df.loc[primary_df['분류'] != '지역', '분류'] = '비지역'
    elif model_type == "secondary":
        primary_df = df[df['분류'] != '지역'].copy()

    label_encoder = {label: i for i, label in enumerate(primary_df['분류'].unique())}
    primary_df['label'] = df['분류'].map(label_encoder)
    return primary_df, label_encoder

def split_data(df: pd.DataFrame, test_size: float = 0.2, stratify_col: str='label') -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_size, stratify=df[stratify_col], random_state=42)