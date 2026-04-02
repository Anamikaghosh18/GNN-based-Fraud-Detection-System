import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             precision_recall_curve, confusion_matrix, 
                             ConfusionMatrixDisplay)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.gnn_model import build_hetero_model
from config import GRAPH_OBJ_PATH 
import mlflow
import mlflow.pytorch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_WEIGHTS = os.path.abspath(os.path.join(CURRENT_DIR, "..", "models", "gnn_fraud_model.pth"))

def find_best_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def visualize_results(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legit', 'Fraud'])
    disp.plot(ax=ax[0], cmap='Blues')
    ax[0].set_title(f'Confusion Matrix (Thresh: {threshold:.2f})')

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ax[1].plot(recall, precision, color='purple', lw=2)
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title('Precision-Recall Curve')
    
   
    save_path = os.path.join(os.path.dirname(MODEL_WEIGHTS), 'evaluation_results.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to: {save_path}")
    plt.show()

def run_inference():
    with mlflow.start_run(run_name="Inference_v1"):
        device = torch.device('cpu')
        
        print(f"🔍 Loading Graph from: {GRAPH_OBJ_PATH}")
        if not os.path.exists(GRAPH_OBJ_PATH):
            print(f"❌ Error: Graph file not found! Check config.py paths.")
            return

        data = torch.load(GRAPH_OBJ_PATH, map_location=device)
        
        # Rebuild and Load Model
        model = build_hetero_model(data.metadata(), hidden_channels=64, out_channels=1)
        
        print(f"🔍 Loading Model Weights from: {MODEL_WEIGHTS}")
        print(f"🔍 Loading Model Weights from: {MODEL_WEIGHTS}")
        if not os.path.exists(MODEL_WEIGHTS):
            print(f"❌ Error: Model weights (.pth) not found at {MODEL_WEIGHTS}")
            return

        # LOAD THE FULL CHECKPOINT
        checkpoint = torch.load(MODEL_WEIGHTS, map_location=device)
        
        # EXTRACT THE STATE DICT
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        
        print("🧠 Inference running...")
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)
            probs = torch.sigmoid(out['transaction']).squeeze().numpy()
            y_true = data['transaction'].y.numpy()
        
        auc = roc_auc_score(y_true, probs)
        ap = average_precision_score(y_true, probs)
        
        print(f"\nResults:\n--- ROC-AUC: {auc:.4f}\n--- PR-AUC: {ap:.4f}")
        
        best_thresh, _ = find_best_threshold(y_true, probs)
        visualize_results(y_true, probs, threshold=best_thresh)


        # Log Parameters
        mlflow.log_param("hidden_channels", 64)
        mlflow.log_param("model_type", "GraphSAGE_Hetero")
        # Log Metrics
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("pr_auc", ap)
        save_path = "models/evaluation_results.png"
        mlflow.log_artifact(save_path)
        mlflow.pytorch.log_model(model, "model")
        
        print("🚀 Run logged to MLflow!")


if __name__ == "__main__":
    run_inference()