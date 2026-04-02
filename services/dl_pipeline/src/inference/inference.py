import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import time

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import mlflow
import mlflow.pytorch

# Paths setup

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_WEIGHTS = os.path.abspath(os.path.join(CURRENT_DIR, "..", "models", "gnn_fraud_model.pth"))
GRAPH_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "..", "data/processed", "graph_data.pt"))
PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..", "..", "..")
)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, '..')))
from models.gnn_model import build_hetero_model


def find_best_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]


def visualize_results(y_true, y_probs, threshold, save_path):
    y_pred = (y_probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legit', 'Fraud'])
    disp.plot(ax=ax[0], cmap='Blues')
    ax[0].set_title(f"Confusion Matrix (thr={threshold:.2f})")

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ax[1].plot(recall, precision)
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision-Recall Curve")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_inference():
    # MLflow setup
    mlflow.set_tracking_uri("file:///" + os.path.join(PROJECT_ROOT, "mlruns").replace("\\", "/"))
    mlflow.set_experiment("GNN_Fraud_Detection")

    device = torch.device("cpu")

    with mlflow.start_run(run_name="inference_run_v1"):
        # Load Data
        if not os.path.exists(GRAPH_PATH):
            raise FileNotFoundError(f"Graph file not found: {GRAPH_PATH}")

        data = torch.load(GRAPH_PATH, map_location=device)

        # Build Model
        model = build_hetero_model(
            data.metadata(),
            hidden_channels=64,
            out_channels=1
        )

   
        # Load Weights
        if not os.path.exists(MODEL_WEIGHTS):
            raise FileNotFoundError(f"Model not found: {MODEL_WEIGHTS}")

        checkpoint = torch.load(MODEL_WEIGHTS, map_location=device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        # Inference
       
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)
            probs = torch.sigmoid(out["transaction"]).squeeze().numpy()
            y_true = data["transaction"].y.numpy()

        # Metrics
        roc_auc = roc_auc_score(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)

        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC : {pr_auc:.4f}")


        # Best Threshold
        best_threshold = find_best_threshold(y_true, probs)

        run_name = f"run_{int(time.time())}"
        run_dir = os.path.join(EXPERIMENTS_DIR, run_name)
        os.makedirs(run_dir, exist_ok=True)

        # Visualization
        plot_path = os.path.join(run_dir, "evaluation_results.png")
        visualize_results(y_true, probs, best_threshold, plot_path)

        # Logging to MLflow 
        mlflow.log_param("model_type", "GraphSAGE_Hetero")
        mlflow.log_param("hidden_channels", 64)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)

        # Log plot
        mlflow.log_artifact(plot_path)
        run_name = f"run_{int(time.time())}"
        run_dir = os.path.join(EXPERIMENTS_DIR, run_name)
        os.makedirs(run_dir, exist_ok=True)

        # Save metrics file
        metrics = {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "best_threshold": float(best_threshold)
        }

        metrics_path = os.path.join(run_dir, "metrics.json")

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        mlflow.log_artifact(metrics_path)

        print("🚀 Experiment logged successfully!")


if __name__ == "__main__":
    run_inference()