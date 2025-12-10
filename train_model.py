import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


EXPERIMENT_NAME = "iris-model-zoo"
MODEL_REGISTRY_NAME = "IrisModel"
MODEL_VERSION = "v1.0.0"
APP_DIR = Path(__file__).parent / "app"
MODEL_PATH = APP_DIR / "model.joblib"
MODEL_META_PATH = APP_DIR / "model_meta.json"
TRACKING_URI = f"file://{Path(__file__).parent.resolve() / 'mlruns'}"


def get_models():
    """Return the collection of models to evaluate."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42
        ),
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=400, multi_class="auto", random_state=42
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        gamma="scale",
                        C=2.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


def compute_metrics(model, X_test, y_test):
    """Compute evaluation metrics and confusion matrix for a fitted model."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
    }

    roc_auc = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, probs, multi_class="ovr")
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        roc_auc = roc_auc_score(y_test, scores, multi_class="ovr")
    if roc_auc is not None:
        metrics["roc_auc"] = roc_auc

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Setosa", "Versicolor", "Virginica"]
    )
    return metrics, cm, report


def plot_confusion_matrix(cm, labels, output_path):
    """Save a confusion matrix plot to disk."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    # Write counts on cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def log_run(model_name, model, X_train, X_test, y_train, y_test):
    """Train a model, compute metrics, and log everything to MLflow."""
    with mlflow.start_run(run_name=model_name) as run, TemporaryDirectory() as tmpdir:
        # Fit model
        model.fit(X_train, y_train)

        # Metrics & artifacts
        metrics, cm, report = compute_metrics(model, X_test, y_test)

        # Log parameters (flatten pipeline if needed)
        if isinstance(model, Pipeline):
            params = {f"{model_name}__{k}": v for k, v in model.named_steps.items()}
        else:
            params = model.get_params()
        mlflow.log_params({k: str(v) for k, v in params.items()})

        # Log metrics
        mlflow.log_metrics(metrics)

        # Tags
        mlflow.set_tags({"version": MODEL_VERSION, "model_name": model_name})

        # Artifacts
        tmpdir_path = Path(tmpdir)
        report_path = tmpdir_path / "classification_report.txt"
        report_path.write_text(report)
        mlflow.log_artifact(report_path)

        cm_path = tmpdir_path / "confusion_matrix.png"
        plot_confusion_matrix(cm, ["Setosa", "Versicolor", "Virginica"], cm_path)
        mlflow.log_artifact(cm_path)

        # Save and log model
        model_path = tmpdir_path / "model.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(model, artifact_path="model")

        return {
            "run_id": run.info.run_id,
            "model_name": model_name,
            "metrics": metrics,
            "fitted_model": model,
        }


def register_best_model(run_id):
    """Register the best model in the MLflow Model Registry."""
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)
    return result.version


def save_best_locally(best_result):
    """Persist the best model and metadata for the Streamlit app."""
    APP_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump(best_result["fitted_model"], MODEL_PATH)

    meta = {
        "best_model": best_result["model_name"],
        "metrics": {
            "accuracy": round(best_result["metrics"].get("accuracy", 0), 3),
            "f1_macro": round(best_result["metrics"].get("f1_macro", 0), 3),
        },
        "mlflow_run_id": best_result["run_id"],
        "version": MODEL_VERSION,
    }
    MODEL_META_PATH.write_text(json.dumps(meta, indent=2))


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    results = []
    for name, model in get_models().items():
        print(f"Training {name}...")
        result = log_run(name, model, X_train, X_test, y_train, y_test)
        results.append(result)

    # Identify best model by F1-macro
    best_result = max(results, key=lambda r: r["metrics"]["f1_macro"])
    print(
        f"Best model: {best_result['model_name']} "
        f"(F1-macro={best_result['metrics']['f1_macro']:.3f})"
    )

    # Register in MLflow Model Registry
    version = register_best_model(best_result["run_id"])
    print(f"Registered {MODEL_REGISTRY_NAME} version {version}")

    # Save locally for the Streamlit app
    save_best_locally(best_result)
    print(f"Saved best model to {MODEL_PATH} and metadata to {MODEL_META_PATH}")


if __name__ == "__main__":
    main()
