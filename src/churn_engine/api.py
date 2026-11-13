"""FastAPI inference endpoint for real-time customer churn scoring."""

from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

from churn_engine.config import ChurnConfig
from churn_engine.preprocessor import ChurnDataPreprocessor, generate_sample_churn_data
from churn_engine.trainer import ChurnModelTrainer

MODEL_PATH = Path("models/churn_model.pkl")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")

app = FastAPI(
    title="Customer Churn Analytics API",
    version="1.0.0",
    description="Real-time customer churn risk scoring and retention action engine.",
)


class CustomerPayload(BaseModel):
    customer_id: str = "CUST-99999"
    tenure_months: int = Field(..., ge=0, le=240, examples=[5])
    monthly_charges: float = Field(..., ge=0.0, examples=[89.5])
    contract_type: str = Field(..., examples=["Month-to-month"])
    support_tickets: int = Field(default=0, ge=0, examples=[3])
    payment_method: str = Field(default="Electronic check", examples=["Electronic check"])


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_tier: Literal["HIGH", "MEDIUM", "LOW"]
    recommended_action: str


def _ensure_artifacts():
    if MODEL_PATH.exists() and PREPROCESSOR_PATH.exists():
        return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)

    cfg = ChurnConfig()
    prep = ChurnDataPreprocessor(cfg)
    df = generate_sample_churn_data(n_samples=600, random_state=42)
    df = prep.clean_data(df)
    df = prep.engineer_features(df)
    X, y = prep.prepare_features(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    trainer = ChurnModelTrainer(cfg)
    trainer.train_and_evaluate(X_train, X_test, y_train, y_test, cv=2)
    joblib.dump(prep, PREPROCESSOR_PATH)
    return joblib.load(MODEL_PATH), prep


@app.get("/health")
def health_check() -> dict:
    return {"status": "healthy", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerPayload) -> PredictionResponse:
    try:
        model_bundle, preprocessor = _ensure_artifacts()
        df = pd.DataFrame([payload.model_dump()])
        df = preprocessor.clean_data(df)
        df = preprocessor.engineer_features(df)
        X, _ = preprocessor.prepare_features(df, fit=False)
        prob = float(model_bundle["model"].predict_proba(X)[0, 1])

        if prob >= 0.70:
            tier, action = "HIGH", "Priority retention outreach + contract upgrade incentive"
        elif prob >= 0.40:
            tier, action = "MEDIUM", "Proactive check-in & loyalty bundle offer"
        else:
            tier, action = "LOW", "Standard lifecycle engagement"

        return PredictionResponse(
            customer_id=payload.customer_id,
            churn_probability=round(prob, 4),
            risk_tier=tier,
            recommended_action=action,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
