from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import mlflow
import pandas as pd
import numpy as np
import config
import pickle
from scripts.training.two_tower_utils import to_tensor
from scripts.preprocessing.prep_utils import preprocess_user_features

# --------------------------------------------
# Initialize app
# --------------------------------------------
app = FastAPI(title="Two Tower Recommendation API")

# --------------------------------------------
# Global objects to keep in memory
# --------------------------------------------
model = None
content_emb_dict = None

# --------------------------------------------
# Request schema
# --------------------------------------------
class UserRequest(BaseModel):
    user_data: dict  # { "deviceid": "xyz", "platform": "Android", ... }
    top_k: int = 50


# --------------------------------------------
# Load model and content embeddings on startup
# --------------------------------------------
@app.on_event("startup")
def load_resources():
    global model, content_emb_dict

    mlflow.set_tracking_uri(config.MODEL_CONFIG['mlruns_dir'])
    model_uri = f"runs:/{config.two_tower_best_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    print(model)

    # Load precomputed content embeddings
    content_emb_dict = torch.load(config.content_emb_path, weights_only=False)

    print(f"✅ Model & content embeddings loaded: {len(content_emb_dict)} items.")


# --------------------------------------------
# Inference helper
# --------------------------------------------
def predict_from_embeddings(user_emb, content_emb):
    with torch.no_grad():
        combined = torch.cat([user_emb, content_emb], dim=1)
        preds = model.output_layer(combined).squeeze(-1)
    return preds.item()  # scalar float


# --------------------------------------------
# API endpoint
# --------------------------------------------
@app.post("/recommend")
def recommend(request: UserRequest):
    if model is None or content_emb_dict is None:
        raise HTTPException(status_code=500, detail="Model not loaded yet.")

    user_dict = request.user_data
    top_k = request.top_k

    # Create a DataFrame for single user (consistent with training format)
    user_df = pd.DataFrame([user_dict])

    # Preprocess using the same encoders used during training
    processed_user = preprocess_user_features(users=user_df, save_dir="artifacts/user_encoders")
    processed_user = processed_user[['platform', 'os_version', 'model', 'networkType', 'district', 'language_selected',
                       'days_since_last_active', 'days_since_signup']]
    # Extract features (assuming first column is 'deviceid')
    user_tensor = processed_user.values
    features = to_tensor(user_tensor.astype('float'))

    # Compute user embedding
    with torch.no_grad():
        user_emb = model.user_tower(features)

    # Compute scores for all content items
    scores = []
    for hashid, content_emb in content_emb_dict.items():
        score = predict_from_embeddings(user_emb, content_emb.unsqueeze(0))
        scores.append((hashid, score))

    # Sort and return top K
    top_items = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    recs = []
    for hid, score in top_items:
        recs.append({
            "hashid": hid,
            "score": score
        })
        
    return {"user_id": user_dict.get("deviceid"), "recommendations": recs}


# --------------------------------------------
# Example curl / test
# --------------------------------------------
# curl -X POST "http://127.0.0.1:8000/recommend" \
#     -H "Content-Type: application/json" \
#     -d '{"user_data": {"deviceid":"U123","platform":"Android","os_version":"13", "model":"Pixel 6", "networkType":"WiFi", "district":"Bangalore", "language_selected":"en", "days_since_last_active":1, "days_since_signup":300}, "top_k":5}'

# uvicorn scripts.postprocessing.realtimeinference:app --reload --port 8000 

# {
#   "user_data": {
#     "deviceid": "197b123e-eb9e-4fc1-a32d-aa86aaea425e",
#     "platform": "ANDROID",
#     "os_version": "13",
#     "model": null,
#     "networkType": "4G",
#     "district": null,
#     "lastknownsubadminarea": null,
#     "language_selected": "en",
#     "created_datetime": "2023-07-11T13:40:05.511Z",
#     "app_updated_at": null,
#     "last_active_at": "2023-07-11T13:40:02.000Z"
#   },
#   "top_k": 5
# }
