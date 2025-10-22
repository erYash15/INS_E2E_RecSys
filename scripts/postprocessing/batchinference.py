import torch
import pandas as pd
import mlflow
import mlflow.pytorch
import os
import pickle
import numpy as np
import config
from scripts.training.two_tower_utils import to_tensor
from scripts.preprocessing.prep_utils import (
    load_data,
    preprocess_user_features,
    preprocess_content_features,
    load_data_without_bad_lines,
)
from tqdm import tqdm

def compute_and_save_user_embeddings(user_df, model, save_path=config.user_emb_path):
    # Split IDs and features
    user_tensor = user_df.values
    device_ids = user_tensor[:, 0].astype(str)
    features = to_tensor(user_tensor[:, 1:].astype('float'))

    # Compute embeddings
    model.eval()
    with torch.no_grad():
        user_emb = model.user_tower(features)

    # Create mapping {device_id: embedding_tensor}
    user_emb_dict = {did: emb for did, emb in zip(device_ids, user_emb)}

    # Save as dictionary
    torch.save(user_emb_dict, save_path)
    print(f"Saved {len(user_emb_dict)} user embeddings to {save_path}")

def compute_and_save_content_embeddings(content_df, model, save_path=config.content_emb_path):
    # Split IDs and features
    content_tensor = content_df.values
    hash_ids = content_tensor[:, 0]#.astype(str)
    features = to_tensor(content_tensor[:, 1:].astype('float'))

    # Compute embeddings
    model.eval()
    with torch.no_grad():
        content_emb = model.content_tower(features)

    # Create mapping {hash_id: embedding_tensor}
    content_emb_dict = {hid: emb for hid, emb in zip(hash_ids, content_emb)}

    # Save as dictionary
    torch.save(content_emb_dict, save_path)
    print(f"Saved {len(content_emb_dict)} content embeddings to {save_path}")

def predict_from_embeddings(user_emb, content_emb, model):
    with torch.no_grad():
        combined = torch.cat([user_emb, content_emb], dim=1)
        preds = model.output_layer(combined).squeeze(-1)
    return preds

if __name__ == "__main__":
    
    # ---------------------
    # Load model
    # ---------------------
    mlflow.set_tracking_uri(config.MODEL_CONFIG['mlruns_dir'])
    model_uri = f"runs:/{config.two_tower_best_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()  # eval mode
    
    print(model)
        
    # ---------------------
    # New Data for Batch Inference
    # ---------------------
    
    users = load_data(config.users)
    processed_users = preprocess_user_features(users = users, save_dir = "artifacts/user_encoders")
    user_tower_cols = ['deviceid', 'platform', 'os_version', 'model', 'networkType', 'district', 'language_selected',
                       'days_since_last_active', 'days_since_signup']
    
    compute_and_save_user_embeddings(processed_users[user_tower_cols], model)

    # Preprocess content features    
    test_content, _ = load_data_without_bad_lines(config.testing)
    test_content = preprocess_content_features(test_content, save_dir= "artifacts/content_encoders")
    print(test_content)
    content_tower_cols = ['hashid', 'newsType', 'newsLanguage', 'sourceName', 'newsDistrict'] + [f'text_emb_{i}' for i in range(128)]
    
    compute_and_save_content_embeddings(test_content[content_tower_cols], model)
    
    print("Batch inference embeddings computation completed.")
    
    user_emb_dict = torch.load(config.user_emb_path, weights_only=False)
    content_emb_dict = torch.load(config.content_emb_path, weights_only=False)
    print(f"Loaded {len(user_emb_dict)} user embeddings from {config.user_emb_path}")
    print(f"Loaded {len(content_emb_dict)} content embeddings from {config.content_emb_path}")
    
    # --- Compute top 50 for each user ---
    results = []

    for user_id, user_emb in tqdm(user_emb_dict.items(), desc="Users", total=len(user_emb_dict)):
        scores = []
        for hashid, content_emb in content_emb_dict.items():
            # Compute prediction score
            score = predict_from_embeddings(user_emb.reshape(1, -1), content_emb.reshape(1, -1), model)
            scores.append((hashid, score))

        # Sort by score (descending) and take top 50
        top_50 = sorted(scores, key=lambda x: x[1], reverse=True)[:50]

        for rank, (hashid, score) in enumerate(top_50, 1):
            results.append({
                "user_id": user_id,
                "hashid": hashid,
                "rank": rank,
                "score": score
            })
            
    # --- Convert to DataFrame ---
    df_top50 = pd.DataFrame(results)
    df_top50.to_csv(config.topk_recommendations_path, index=False)

    print(f"Saved top 50 recommendations for each user to {config.topk_recommendations_path}")