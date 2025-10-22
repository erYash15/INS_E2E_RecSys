import os

import mlflow
import mlflow.pytorch
import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from config import MODEL_CONFIG
from scripts.training.two_tower_utils import create_dataloaders, load_data, to_tensor


class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, content_dim, embedding_dim, dropout=0.2):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )
        self.content_tower = nn.Sequential(
            nn.Linear(content_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(embedding_dim * 2, 1)  # regression

    def forward(self, u, c):
        u_vec = self.user_tower(u)
        c_vec = self.content_tower(c)
        combined = torch.cat([u_vec, c_vec], dim=1)
        out = self.output_layer(combined)
        return out.squeeze(-1), u_vec, c_vec


def objective(
    trial, user_dim, content_dim, tX_user, tX_content, ty, vX_user, vX_content, vy
):
    # Sample hyperparameters
    embedding_dim = trial.suggest_categorical(
        "embedding_dim", MODEL_CONFIG["hpo_params"]["embedding_dim"]
    )
    dropout = trial.suggest_float(
        "dropout", *MODEL_CONFIG["hpo_params"]["dropout_range"]
    )
    lr = trial.suggest_float("lr", *MODEL_CONFIG["hpo_params"]["lr_range"], log=True)
    batch_size = trial.suggest_categorical(
        "batch_size", MODEL_CONFIG["hpo_params"]["batch_size"]
    )

    # Dataloaders
    train_loader, val_loader = create_dataloaders(
        tX_user, tX_content, ty, vX_user, vX_content, vy, batch_size
    )

    # MLflow tracking
    os.makedirs(MODEL_CONFIG["mlruns_dir"], exist_ok=True)
    mlflow.set_tracking_uri(f"file:{MODEL_CONFIG['mlruns_dir']}")
    mlflow.set_experiment(MODEL_CONFIG["experiment_name"])

    with mlflow.start_run(run_name=f"optuna_trial_{trial.number}"):
        mlflow.log_params(
            {
                "embedding_dim": embedding_dim,
                "dropout": dropout,
                "lr": lr,
                "batch_size": batch_size,
            }
        )

        model = TwoTowerModel(user_dim, content_dim, embedding_dim, dropout)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Training loop
        for epoch in range(MODEL_CONFIG["num_epochs"]):
            model.train()
            train_loss = 0
            for u, c, yb in train_loader:
                optimizer.zero_grad()
                preds, _, _ = model(u, c)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for u, c, yb in val_loader:
                    preds, _, _ = model(u, c)
                    val_loss += criterion(preds, yb).item()
            avg_val_loss = val_loss / len(val_loader)

            # Log metrics
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

        # Log model
        mlflow.pytorch.log_model(model, "model")

        trial.set_user_attr("run_id", mlflow.active_run().info.run_id)
        return avg_val_loss


# Section 3: Run HPO Study
if __name__ == "__main__":
    # Load and preprocess data
    data = load_data(MODEL_CONFIG["data_path"])
    tX_user, tX_content, ty = (
        to_tensor(data["tX_user"]),
        to_tensor(data["tX_content"]),
        to_tensor(data["ty"]),
    )
    vX_user, vX_content, vy = (
        to_tensor(data["vX_user"]),
        to_tensor(data["vX_content"]),
        to_tensor(data["vy"]),
    )

    user_dim, content_dim = tX_user.shape[1], tX_content.shape[1]

    # Run Optuna HPO
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(
            trial,
            user_dim,
            content_dim,
            tX_user,
            tX_content,
            ty,
            vX_user,
            vX_content,
            vy,
        ),
        n_trials=MODEL_CONFIG["num_trials"],
    )

    # Print best trial
    best_trial = study.best_trial
    print(f"Best Validation Loss: {best_trial.value:.6f}")
    print("Best Hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")
    print(f"Best MLflow run_id: {best_trial.user_attrs['run_id']}")
