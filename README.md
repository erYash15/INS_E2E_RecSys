🚀 Overview

INS_E2E_RecSys is a production-grade recommendation system built using the Two-Tower (Dual Encoder) architecture, demonstrating a full MLOps workflow.

It combines preprocessing, training, inferencing , API deployment, jenkins orchestration and respective pipelines using industry-standard tools like MLflow, Prefect, and Jenkins.

While the model architecture is intentionally kept simple for demonstration, the pipeline structure and orchestration mimic real-world production systems.

🎯 Table of Content

The goal is to build a personalized recommendation engine that connects users with the most relevant content based on behavioral and contextual features.

Key Features

Modularized, production-grade project structure.

Config-driven ML pipeline using Prefect.

Automated orchestration with Jenkins.

Experiment tracking and model versioning via MLflow.

Support for local SageMaker pipeline simulation.

Extensible for scaling to cloud environments.

📊 Problem Statement

Build a Two-Tower Recommendation System that:

Learns user and content embeddings separately.

Computes similarity scores to recommend top-K content per user.

Utilizes real-world data signals like user behavior, device info, and content metadata.

💼 Business Problem

Personalized recommendations drive engagement, retention, and revenue.
This project demonstrates how an organization can:

Leverage behavioral + contextual data.

Automate data-to-deployment workflows.

Track experiments efficiently for model reproducibility.

⚙️ Architecture Overview

Core Components:

Preprocessing – Handles encoding, feature transformations, and missing data.

Model Training – Implements a Two-Tower model using PyTorch.

Evaluation – Computes metrics like Hit@K, NDCG@K, and MAP.

Post-Processing – Generates and stores top-K recommendations.

Pipeline Orchestration – Managed by Prefect and Jenkins.

Experiment Tracking – All trials logged to MLflow UI.

🧩 Comparative Study: Two-Tower vs Other Architectures
Aspect	Two-Tower Model	Matrix Factorization	Transformer-based RecSys
Scalability	✅ Highly scalable (independent towers)	⚠️ Limited for large datasets	❌ Computationally expensive
Cold Start Handling	✅ Works with metadata features	❌ Needs user history	✅ Handles with embeddings
Online Inference Speed	⚡ Fast (precomputed embeddings)	🐢 Slower due to dense similarity lookup	⚡ Moderate
Interpretability	⚠️ Moderate	✅ High	⚠️ Lower
Use Case Fit	✅ Large-scale personalized systems	⚠️ Small datasets	✅ Context-aware systems

Conclusion:
The Two-Tower model offers the best trade-off between scalability, simplicity, and performance, making it ideal for real-world recommender systems.

🔧 Technologies Used
Category	Tools
Language	Python 3.10
Frameworks	PyTorch, Scikit-learn
Pipeline Orchestration	Prefect
Automation & CI/CD	Jenkins
Experiment Tracking	MLflow
Visualization	Matplotlib, Seaborn
Environment Management	pyenv, virtualenv
Deployment Simulation	SageMaker Local Mode
🧮 Evaluation Metrics

Hit@K – Measures whether relevant items appear in the top-K list.

NDCG@K – Captures ranking quality considering position.

MAP (Mean Average Precision) – Averages precision across users.

🧠 Pipeline Flow

Prefect Pipeline (flow.py)

Loads raw data → preprocesses → trains model → logs metrics.

Jenkins Orchestration

Automates the Prefect flow on commits or nightly runs.

MLflow Tracking

Logs hyperparameters, metrics, and models for every experiment.

Post-processing

Generates top-K recommendations and stores artifacts.

Deployment

Uses SageMaker local simulation or can easily extend to AWS deployment.

🖥️ MLflow UI

To visualize runs, metrics, and models:

cd artifacts
mlflow ui


Then open your browser and navigate to:

http://127.0.0.1:5000

You’ll see:

Trial details (0th, 1st, …)

Hyperparameters

Training & validation loss curves

Complete trial list with performance comparisons.

🧰 How to Run
1️⃣ Create Environment
pyenv virtualenv 3.10.13 ins_venv
pyenv activate ins_venv
pip install -r requirements.txt

2️⃣ Run Prefect Pipeline
python flows/train_pipeline.py

3️⃣ Trigger from Jenkins

Configure Jenkins job with:

python flows/train_pipeline.py

4️⃣ Launch MLflow UI
mlflow ui --port 5000

📦 Directory Structure
INS_E2E_RecSys/
│
├── config.py
├── experiments/*
├── scripts/
│   ├── preprocessing/
│   │   ├── prep_utils.py
│   │   └── preprocessing.py
│   ├── training/
│   │   └── two_tower_utils.py
│   │   └── two_tower.py
│   └── postprocessing/
│       └── evaluation.py
│       └── batchinference.py
│       └── batchinference.py
│
├── flows/
│   └── train_pipeline.py
│
├── artifacts/
│   ├── models/
│   ├── user_encoders/
│   ├── content_encoders/
│   └── mlruns/
│
├── experiments/
│
├── Jenkinsfile
└── README.md

🧩 Future Improvements

Add transformer-based hybrid retrieval.

Integrate feature store (e.g., Feast).

Support for online A/B testing.

Optimize for real-time recommendations.

🗓️ Author

Yash Gupta
Data Scientist | MLOps Enthusiast | Forecasting & Recommendation Systems

Would you like me to also generate a PowerPoint summary (3–4 slides) from this README (Problem → Approach → Architecture → Results)? It’ll be perfect for presentation/demo.
