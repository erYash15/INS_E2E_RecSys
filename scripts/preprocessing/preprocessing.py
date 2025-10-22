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

if __name__ == "__main__":

    # --------------------------
    # 1. User Tower Feature Engineering
    # --------------------------
    users = load_data(config.users)
    processed_users = preprocess_user_features(
        users=users, save_dir="artifacts/user_encoders"
    )

    print("User features preprocessing completed.")

    # --------------------------
    # 2. Content Tower Feature Engineering
    # --------------------------

    train_df, train_lines = load_data_without_bad_lines(config.training)
    test_df, test_lines = load_data_without_bad_lines(config.testing)

    # Calculate percentage of bad lines (excluding header)
    train_bad_pct = ((train_lines - 1 - len(train_df)) / (train_lines - 1)) * 100
    test_bad_pct = ((test_lines - 1 - len(test_df)) / (test_lines - 1)) * 100

    print(f"Bad lines in train: {train_bad_pct:.2f}%")
    print(f"Bad lines in test: {test_bad_pct:.2f}%")

    # Split train into train and validation sets for contents only as asked
    train_X, val_X = train_test_split(train_df, test_size=0.2, random_state=42)

    print(f"Train shape: {train_X.shape}, Validation shape: {val_X.shape}")

    train_X = preprocess_content_features(
        train_df=train_X, save_dir="artifacts/content_encoders"
    )
    val_X = preprocess_content_features(
        train_df=val_X, save_dir="artifacts/content_encoders"
    )

    print("Content features preprocessing completed.")

    # --------------------------
    # 3. Interaction / Label Engineering
    # --------------------------

    events = pd.read_csv(os.path.join(config.events_list[0]))
    # Concatenate vertically (row-wise)
    for i in range(1, len(config.events_list)):
        temp_df = pd.read_csv(os.path.join(config.events_list[i]))
        events = pd.concat([events, temp_df], ignore_index=True)

    tX_user, tX_content, ty = create_training_data(
        events=events, user_features=processed_users, content_features=train_X
    )
    vX_user, vX_content, vy = create_training_data(
        events=events, user_features=processed_users, content_features=val_X
    )

    # Save training and validation data using pickle

    data_to_save = {
        "tX_user": tX_user,
        "tX_content": tX_content,
        "ty": ty,
        "vX_user": vX_user,
        "vX_content": vX_content,
        "vy": vy,
    }

    with open("artifacts/train_val_data.pkl", "wb") as f:
        pickle.dump(data_to_save, f)

    print("Training and validation data saved to train_val_data.pkl")
