"""Model training, SMOTE balancing, MLflow tracking, and artifact persistence."""

from pathlib import Path
from typing import Dict, Tuple
import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV

from churn_engine.config import ChurnConfig, setup_logging

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = setup_logging()


class ChurnModelTrainer:
    """Trains, tunes, evaluates, and logs Logistic Regression & Random Forest models."""

    def __init__(self, config: ChurnConfig):
        self.config = config
        self.models: dict = {}
        self.best_model = None
        self.best_model_name: str | None = None
        self.metrics: dict = {}
        Path("visualizations").mkdir(parents=True, exist_ok=True)
        Path("reports").mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)

    def handle_imbalance(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Balance minority churn class via SMOTE when imbalance ratio < 0.5."""
        class_dist = y_train.value_counts()
        imbalance_ratio = class_dist.min() / class_dist.max()
        if imbalance_ratio < 0.5:
            smote = SMOTE(
                random_state=self.config.get("random_state", 42), sampling_strategy="auto"
            )
            return smote.fit_resample(X_train, y_train)
        return X_train, y_train

    def train_logistic_regression(
        self, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 3
    ) -> None:
        param_grid = {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
            "class_weight": ["balanced", None],
        }
        lr = LogisticRegression(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(lr, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        grid_search.fit(X_train, y_train)
        self.models["Logistic Regression"] = grid_search.best_estimator_
        mlflow.log_params({f"lr_{k}": v for k, v in grid_search.best_params_.items()})
        mlflow.log_metric("lr_cv_roc_auc", grid_search.best_score_)

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 3) -> None:
        param_grid = {
            "n_estimators": [100, 150],
            "max_depth": [8, 16, None],
            "min_samples_split": [2, 5],
            "class_weight": ["balanced", None],
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        grid_search.fit(X_train, y_train)
        self.models["Random Forest"] = grid_search.best_estimator_
        mlflow.log_params({f"rf_{k}": v for k, v in grid_search.best_params_.items()})
        mlflow.log_metric("rf_cv_roc_auc", grid_search.best_score_)

    def evaluate_model(
        self, model_name: str, model, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "churn_precision": float(report.get("1", {}).get("precision", 0.0)),
            "churn_recall": float(report.get("1", {}).get("recall", 0.0)),
            "churn_f1": float(report.get("1", {}).get("f1-score", 0.0)),
        }
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"{model_name.lower().replace(' ', '_')}_{metric_name}", value)

        self._plot_confusion_matrix(confusion_matrix(y_test, y_pred), model_name)
        self._plot_roc_curve(y_test, y_pred_proba, model_name)
        return metrics

    def _plot_confusion_matrix(self, cm, model_name: str) -> None:
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Churn", "Churn"],
            yticklabels=["Not Churn", "Churn"],
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        filepath = f"visualizations/{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        plt.savefig(filepath, bbox_inches="tight", dpi=150)
        plt.close()
        mlflow.log_artifact(filepath)

    def _plot_roc_curve(self, y_test, y_pred_proba, model_name: str) -> None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        filepath = f"visualizations/{model_name.lower().replace(' ', '_')}_roc_curve.png"
        plt.savefig(filepath, bbox_inches="tight", dpi=150)
        plt.close()
        mlflow.log_artifact(filepath)

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, cv: int = 3) -> Dict:
        mlflow.set_experiment(self.config.get("experiment_name", "Churn_Prediction"))
        with mlflow.start_run(run_name="Churn_Model_Training"):
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])

            X_train_bal, y_train_bal = self.handle_imbalance(X_train, y_train)
            self.train_logistic_regression(X_train_bal, y_train_bal, cv=cv)
            self.train_random_forest(X_train_bal, y_train_bal, cv=cv)

            for model_name, model in self.models.items():
                self.metrics[model_name] = self.evaluate_model(model_name, model, X_test, y_test)

            best_name = max(self.metrics, key=lambda k: self.metrics[k]["roc_auc"])
            self.best_model_name = best_name
            self.best_model = self.models[best_name]
            mlflow.log_param("best_model", best_name)

            if best_name == "Random Forest":
                self._analyze_feature_importance(X_train)
            else:
                self._analyze_coefficients(X_train)

            self._save_model()
        return self.metrics

    def _analyze_feature_importance(self, X_train: pd.DataFrame) -> None:
        importance_df = pd.DataFrame(
            {"feature": X_train.columns, "importance": self.best_model.feature_importances_}
        ).sort_values("importance", ascending=False)
        importance_df.to_csv("reports/feature_importance.csv", index=False)
        mlflow.log_artifact("reports/feature_importance.csv")

    def _analyze_coefficients(self, X_train: pd.DataFrame) -> None:
        coefs = self.best_model.coef_[0]
        coef_df = pd.DataFrame(
            {"feature": X_train.columns, "coefficient": coefs, "abs_coefficient": np.abs(coefs)}
        ).sort_values("abs_coefficient", ascending=False)
        coef_df.to_csv("reports/model_coefficients.csv", index=False)
        mlflow.log_artifact("reports/model_coefficients.csv")

    def _save_model(self) -> None:
        model_data = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "metrics": self.metrics[self.best_model_name],
        }
        joblib.dump(model_data, "models/churn_model.pkl")
        mlflow.sklearn.log_model(self.best_model, artifact_path="model")
