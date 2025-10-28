import os

import joblib
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import config

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a DataFrame."""
    return pd.read_csv(file_path)


def load_data_without_bad_lines(path):
    """Load data from CSV while counting total lines and skipping bad lines."""
    with open(path, encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    df = pd.read_csv(path, on_bad_lines="skip")
    return df, total_lines


def preprocess_user_features(
    users: pd.DataFrame, save_dir: str = "user_encoders"
) -> pd.DataFrame:
    """Encode categorical and time-based user features and save LabelEncoders."""
    user_features = users.copy()

    # Create directory to save encoders if not exist
    os.makedirs(save_dir, exist_ok=True)

    # Define categorical columns to encode
    categorical_user_cols = [
        "platform",
        "os_version",
        "model",
        "networkType",
        "district",
        "language_selected",
    ]

    # Encode categorical user features
    for col in categorical_user_cols:
        encoder_path = os.path.join(save_dir, f"user_{col}_encoder.pkl")
        # Check if encoder exists
        if os.path.isfile(encoder_path):
            # Load existing encoder
            le = joblib.load(encoder_path)
        else:
            # Fit new encoder and save
            le = LabelEncoder()
            le.fit(user_features[col].astype(str))
            joblib.dump(le, encoder_path)
        # Apply encoder to column
        user_features[col] = (
            user_features[col]
            .astype(str)
            .apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        )

    # Convert timestamps to UTC
    user_features["last_active_at"] = pd.to_datetime(
        user_features["last_active_at"], utc=True, errors="coerce"
    )
    user_features["created_datetime"] = pd.to_datetime(
        user_features["created_datetime"], utc=True, errors="coerce"
    )

    # Current UTC time
    now = pd.Timestamp.now(tz="UTC")

    # Create recency features
    user_features["days_since_last_active"] = (
        now - user_features["last_active_at"]
    ).dt.days
    user_features["days_since_signup"] = (
        now - user_features["created_datetime"]
    ).dt.days

    # Fill missing values with 0
    return user_features.fillna(0)


def preprocess_content_features(
    train_df: pd.DataFrame,
    save_dir: str = "content_encoders",
    max_features: int = 5000,
    n_svd: int = 128,
):
    """Encode categorical and text features for content; save/reuse encoders, TF-IDF, and SVD."""
    os.makedirs(save_dir, exist_ok=True)
    categorical_content_cols = [
        "newsType",
        "newsLanguage",
        "sourceName",
        "newsDistrict",
    ]
    content_features = train_df.copy()

    # Encode categorical columns
    for col in categorical_content_cols:
        encoder_path = os.path.join(save_dir, f"content_{col}_encoder.pkl")
        if os.path.isfile(encoder_path):
            le = joblib.load(encoder_path)
        else:
            le = LabelEncoder()
            le.fit(content_features[col].astype(str))
            joblib.dump(le, encoder_path)
        content_features[col] = (
            content_features[col]
            .astype(str)
            .apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        )

    # Prepare text: title*3 + content
    content_features["text"] = (
        content_features["title"].fillna("") * 3
        + " "
        + content_features["content"].fillna("")
    )

    # TF-IDF Vectorizer
    tfidf_path = os.path.join(save_dir, "tfidf_vectorizer.pkl")
    if os.path.isfile(tfidf_path):
        tfidf = joblib.load(tfidf_path)
        text_features = tfidf.transform(content_features["text"])
    else:
        tfidf = TfidfVectorizer(max_features=max_features)
        text_features = tfidf.fit_transform(content_features["text"])
        joblib.dump(tfidf, tfidf_path)

    # Truncated SVD for dimensionality reduction
    svd_path = os.path.join(save_dir, "svd_model.pkl")
    if os.path.isfile(svd_path):
        svd = joblib.load(svd_path)
        text_emb = svd.transform(text_features)
    else:
        svd = TruncatedSVD(n_components=n_svd, random_state=42)
        text_emb = svd.fit_transform(text_features)
        joblib.dump(svd, svd_path)

    # Concatenate embeddings with original dataframe
    text_emb_df = pd.DataFrame(
        text_emb, columns=[f"text_emb_{i}" for i in range(text_emb.shape[1])]
    )
    content_features = pd.concat(
        [content_features.reset_index(drop=True), text_emb_df], axis=1
    )

    return content_features.fillna(0)


def create_training_data(
    events: pd.DataFrame, user_features: pd.DataFrame, content_features: pd.DataFrame
):
    """Assign engagement scores and merge only selected content events with user/content features."""

    selected_hashes = list(content_features["hashid"].unique())
    selected_devices = list(user_features["deviceid"].unique())
    # Filter events to only include selected hashIds
    events = events[events["hashId"].isin(selected_hashes)].copy()
    events = events[events["deviceId"].isin(selected_devices)].copy()

    # Assign engagement scores to event types
    event_weights = {
        "TimeSpent-Front": 0.3,
        "TimeSpent-Back": 0.5,
        "News Bookmarked": 1.0,
        "News Shared": 1.0,
    }
    events["engagement_score"] = events["event_type"].map(event_weights).fillna(0)

    # Convert timestamp to UTC
    events["eventTimestamp"] = pd.to_datetime(
        events["eventTimestamp"], unit="ms", utc=True
    )

    # Merge with user features
    train_df = events.merge(
        user_features,
        left_on="deviceId",
        right_on="deviceid",
        how="left",
        suffixes=("", "_user"),
    )
    # Merge with content features
    train_df = train_df.merge(
        content_features,
        left_on="hashId",
        right_on="hashid",
        how="left",
        suffixes=("", "_content"),
    )

    # User and content tower columns
    user_tower_cols = ["deviceId"] + config.user_tower_cols
    content_tower_cols = ["hashId"] + config.content_tower_cols

    # Target
    target_col = "engagement_score"

    # Split features and target
    X_user = train_df[user_tower_cols]
    X_content = train_df[content_tower_cols]
    y = train_df[target_col]

    return X_user, X_content, y
