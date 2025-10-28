import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

import config
from scripts.preprocessing.prep_utils import (
    create_training_data,
    load_data,
    load_data_without_bad_lines,
    preprocess_content_features,
    preprocess_user_features,
)


def preprocess_users():
    """Load and preprocess user features."""
    users = load_data(config.users)
    processed_users = preprocess_user_features(
        users=users, save_dir="artifacts/user_encoders"
    )
    print("✅ User features preprocessing completed.")
    return processed_users


def preprocess_contents():
    """Load, clean, split, and preprocess content features."""
    # Load train and test data
    train_df, train_lines = load_data_without_bad_lines(config.training)
    test_df, test_lines = load_data_without_bad_lines(config.testing)

    # Calculate bad line percentages
    train_bad_pct = ((train_lines - 1 - len(train_df)) / (train_lines - 1)) * 100
    test_bad_pct = ((test_lines - 1 - len(test_df)) / (test_lines - 1)) * 100
    print(f"⚠️ Bad lines in train: {train_bad_pct:.2f}%, test: {test_bad_pct:.2f}%")

    # Split into train/val
    train_X, val_X = train_test_split(train_df, test_size=0.2, random_state=42)
    print(f"📊 Train shape: {train_X.shape}, Validation shape: {val_X.shape}")

    # Preprocess
    train_X = preprocess_content_features(
        train_df=train_X, save_dir="artifacts/content_encoders"
    )
    val_X = preprocess_content_features(
        train_df=val_X, save_dir="artifacts/content_encoders"
    )

    print("✅ Content features preprocessing completed.")
    return train_X, val_X


def load_all_events():
    """Load and concatenate all event files."""
    print("📂 Loading event files...")
    events = pd.read_csv(config.events_list[0])
    for path in config.events_list[1:]:
        temp_df = pd.read_csv(path)
        events = pd.concat([events, temp_df], ignore_index=True)
    print(f"✅ Loaded {len(events)} total event records.")
    return events


def create_and_save_train_val_data(processed_users, train_X, val_X, events):
    """Create interaction training data and save to pickle."""
    print("🔧 Creating training and validation data...")
    tX_user, tX_content, ty = create_training_data(
        events=events, user_features=processed_users, content_features=train_X
    )
    vX_user, vX_content, vy = create_training_data(
        events=events, user_features=processed_users, content_features=val_X
    )

    data_to_save = {
        "tX_user": tX_user,
        "tX_content": tX_content,
        "ty": ty,
        "vX_user": vX_user,
        "vX_content": vX_content,
        "vy": vy,
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/train_val_data.pkl", "wb") as f:
        pickle.dump(data_to_save, f)

    print("✅ Training and validation data saved to artifacts/train_val_data.pkl")


def main():
    """Main flow for preprocessing and data preparation."""
    print("🚀 Starting preprocessing pipeline...")
    processed_users = preprocess_users()
    train_X, val_X = preprocess_contents()
    events = load_all_events()
    create_and_save_train_val_data(processed_users, train_X, val_X, events)
    print("🎉 Preprocessing Pipeline completed successfully!")


if __name__ == "__main__":
    main()
