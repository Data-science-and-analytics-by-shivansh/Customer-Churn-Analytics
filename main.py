"""CLI entrypoint orchestrating training, evaluation, business impact, and serving."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import joblib  # noqa: E402
import uvicorn  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from churn_engine.config import ChurnConfig, setup_logging  # noqa: E402
from churn_engine.impact import BusinessImpactAnalyzer  # noqa: E402
from churn_engine.preprocessor import ChurnDataPreprocessor  # noqa: E402
from churn_engine.trainer import ChurnModelTrainer  # noqa: E402


def run_pipeline(config_path: str = "config/churn_config.yaml") -> dict:
    config = ChurnConfig(config_path)
    logger = setup_logging(config.get("log_level", "INFO"))

    preprocessor = ChurnDataPreprocessor(config)
    trainer = ChurnModelTrainer(config)
    impact_analyzer = BusinessImpactAnalyzer(config)

    data_path = config.get("data_path", "data/churn_data.csv")
    df = preprocessor.load_data(data_path)
    df = preprocessor.clean_data(df)
    df = preprocessor.engineer_features(df)

    X, y = preprocessor.prepare_features(df, fit=True)
    test_size = config.get("test_size", 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    y_pred_proba = trainer.best_model.predict_proba(X_test)[:, 1]
    impact = impact_analyzer.calculate_retention_improvement(y_test, y_pred_proba)

    joblib.dump(preprocessor, "models/preprocessor.pkl")
    logger.info("=== TRAINING COMPLETE ===")
    return {
        "best_model": trainer.best_model_name,
        "metrics": trainer.metrics[trainer.best_model_name],
        "business_impact": impact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise Customer Churn Analytics Pipeline")
    parser.add_argument(
        "--config", type=str, default="config/churn_config.yaml", help="Path to YAML config"
    )
    parser.add_argument("--mode", choices=["train", "serve"], default="train")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.mode == "train":
        result = run_pipeline(args.config)
        print(json.dumps(result, indent=2))
    else:
        uvicorn.run("churn_engine.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
