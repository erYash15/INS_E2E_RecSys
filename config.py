users = "test_data/devices/part-00000-cdb2cdd7-9d14-4000-b947-4d0475444217-c000.csv"
events_list = [
    "test_data/event/part-00000-7e210b01-29d8-430f-988f-d3e3b34da614-c000.csv",
    "test_data/event/part-00001-7e210b01-29d8-430f-988f-d3e3b34da614-c000.csv",
    "test_data/event/part-00002-7e210b01-29d8-430f-988f-d3e3b34da614-c000.csv",
    "test_data/event/part-00003-7e210b01-29d8-430f-988f-d3e3b34da614-c000.csv"

]
training = "test_data/training_content/part-00000-a34a1545-5cf1-47b9-93c2-29c1d3f0bfb7-c000.csv"
testing = "test_data/testing_content/part-00000-8be13c58-b74d-4e30-8877-c8b5e168035a-c000.csv"

MODEL_CONFIG = {
    "data_path": "artifacts/train_val_data.pkl",
    "mlruns_dir": "artifacts/mlruns",
    "experiment_name": "two_tower_recommender",
    "num_trials": 10,
    "num_epochs": 10,
    "hpo_params": {
        "embedding_dim": [32, 64, 96],
        "dropout_range": (0.1, 0.3),
        "lr_range": (1e-4, 1e-2),
        "batch_size": [1024, 2048]
    }
}

two_tower_best_id = "b0caa2b2421e4d4e80179cb9b29faff6"