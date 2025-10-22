import torch
import pandas as pd
import mlflow
import mlflow.pytorch
import os
import pickle
import numpy as np
import config
from scripts.training.two_tower_utils import to_tensor



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
    # Load training and validation data using pickle
    # ---------------------
    with open(config.MODEL_CONFIG['data_path'], "rb") as f:
        data = pickle.load(f)
        tX_user = data["tX_user"]
        tX_content = data["tX_content"]
        ty = data["ty"]
        vX_user = data["vX_user"]
        vX_content = data["vX_content"]
        vy = data["vy"]

        print("✅ Data loaded successfully from pickle")