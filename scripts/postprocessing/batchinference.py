import os

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from tqdm import tqdm

import config
from scripts.preprocessing.prep_utils import (
    load_data,
    load_data_without_bad_lines,
    preprocess_content_features,
    preprocess_user_features,
)
from scripts.training.two_tower_utils import to_tensor


# -----------------------------
# Embedding utility functions
# -----------------------------
def compute_and_save_user_embeddings(user_df, model, save_path):
    user_tensor = user_df.values
    device_ids = user_tensor[:, 0].astype(str)
    features = to_tensor(user_tensor[:, 1:].astype("float"))

    model.eval()
    with torch.no_grad():
        user_emb = model.user_tower(features)

    user_emb_dict = {did: emb for did, emb in zip(device_ids, user_emb)}
    torch.save(user_emb_dict, save_path)
    print(f"✅ Saved {len(user_emb_dict)} user embeddings → {save_path}")


def compute_and_save_content_embeddings(content_df, model, save_path):
    content_tensor = content_df.values
    hash_ids = content_tensor[:, 0]
    features = to_tensor(content_tensor[:, 1:].astype("float"))

    model.eval()
    with torch.no_grad():
        content_emb = model.content_tower(features)

    content_emb_dict = {hid: emb for hid, emb in zip(hash_ids, content_emb)}
    torch.save(content_emb_dict, save_path)
    print(f"✅ Saved {len(content_emb_dict)} content embeddings → {save_path}")


def predict_from_embeddings(user_emb, content_emb, model):
    with torch.no_grad():
        combined = torch.cat([user_emb, content_emb], dim=1)
        preds = model.output_layer(combined).squeeze(-1)
    return preds


# -----------------------------
# Main function (parameterized)
# -----------------------------
def run_batch_inference(
    mlruns_dir,
    best_run_id,
    users,
    test_content,
    output_dir="artifacts",
    topk=50,
):
    """
    Runs batch inference to compute user & content embeddings,
    and generates top-K recommendations.
    """

    os.makedirs(output_dir, exist_ok=True)
    user_emb_path = os.path.join(output_dir, "user_embeddings.pt")
    content_emb_path = os.path.join(output_dir, "content_embeddings.pt")

    # --- Load model from MLflow ---
    mlflow.set_tracking_uri(mlruns_dir)
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    print(f"📦 Loaded model from MLflow run {best_run_id}")

    # --- Preprocess Users ---

    processed_users = preprocess_user_features(
        users, save_dir=f"{output_dir}/user_encoders"
    )
    user_tower_cols = [
        "deviceid",
        "platform",
        "os_version",
        "model",
        "networkType",
        "district",
        "language_selected",
        "days_since_last_active",
        "days_since_signup",
    ]
    compute_and_save_user_embeddings(
        processed_users[user_tower_cols], model, user_emb_path
    )

    # --- Preprocess Content ---

    test_content = preprocess_content_features(
        test_content, save_dir=f"{output_dir}/content_encoders"
    )
    content_tower_cols = [
        "hashid",
        "newsType",
        "newsLanguage",
        "sourceName",
        "newsDistrict",
    ] + [f"text_emb_{i}" for i in range(128)]
    compute_and_save_content_embeddings(
        test_content[content_tower_cols], model, content_emb_path
    )

    # --- Load embeddings ---
    user_emb_dict = torch.load(user_emb_path, weights_only=False)
    content_emb_dict = torch.load(content_emb_path, weights_only=False)
    print(
        f"📊 Loaded {len(user_emb_dict)} user embeddings, {len(content_emb_dict)} content embeddings"
    )

    # --- Compute top-K recommendations ---
    results = []
    for user_id, user_emb in tqdm(
        user_emb_dict.items(), desc="Scoring users", total=len(user_emb_dict)
    ):
        scores = []
        for hashid, content_emb in content_emb_dict.items():
            score = predict_from_embeddings(
                user_emb.reshape(1, -1), content_emb.reshape(1, -1), model
            )
            scores.append((hashid, score.item()))

        top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:topk]
        for rank, (hashid, score) in enumerate(top_k, 1):
            results.append(
                {"user_id": user_id, "hashid": hashid, "rank": rank, "score": score}
            )

    # --- Save results ---
    df_topk = pd.DataFrame(results)
    return df_topk


# -----------------------------
# CLI entry point
# -----------------------------
if __name__ == "__main__":
    users = load_data(config.users)
    test_content, _ = load_data_without_bad_lines(config.testing)
    df = run_batch_inference(
        mlruns_dir=config.MODEL_CONFIG["mlruns_dir"],
        best_run_id=config.two_tower_best_id,
        users=users,
        test_content=test_content,
        output_dir="artifacts",
        topk=50,
    )
    topk = 50
    topk_csv = os.path.join("artifacts", f"top{topk}_recommendations.csv")
    df.to_csv(topk_csv, index=False)
    print(f"🎯 Saved top-{topk} recommendations → {topk_csv}")
