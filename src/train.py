# src/train.py

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os
from mlflow.models.signature import infer_signature
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# point MLflow calls to the running server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# set experiment name (will create it automatically if it doesn't exist)
mlflow.set_experiment("mlops-assignment-1")

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dataset loaded successfully")

# Ensure models folder exists
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Function to train, evaluate, log with MLflow
def train_and_log(model, model_name, params={}):
    with mlflow.start_run(run_name=model_name):
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # === New Part: Save evaluation results as CSV and log as artifact ===
        import pandas as pd
        eval_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
        eval_csv = f"results/{model_name}_eval.csv"
        eval_df.to_csv(eval_csv, index=False)
        mlflow.log_artifact(eval_csv, artifact_path="evaluation")

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        print(f"\n🔹 {model_name} Results:")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Save model locally
        model_path = f"models/{model_name}_model.pkl"
        joblib.dump(model, model_path)
        print(f" {model_name} model saved at {model_path}")

        # Log params & metrics
        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1})

        # === New Part: Add signature + input_example ===
        input_example = X_train[:2]
        signature = infer_signature(X_train, model.predict(X_train[:2]))

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        # Log confusion matrix as artifact
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        plt.title(f"{model_name} - Confusion Matrix")
        cm_path = f"results/{model_name}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # Log the saved .pkl model file too
        mlflow.log_artifact(model_path, artifact_path="saved_models")

# 4. Logistic Regression
train_and_log(
    LogisticRegression(max_iter=200),
    "LogisticRegression",
    {"max_iter": 200}
)

# 5. Random Forest
train_and_log(
    RandomForestClassifier(n_estimators=100, random_state=42),
    "RandomForest",
    {"n_estimators": 100, "random_state": 42}
)

# 6. SVM
train_and_log(
    SVC(kernel="linear", probability=True, random_state=42),
    "SVM",
    {"kernel": "linear", "probability": True, "random_state": 42}
)
