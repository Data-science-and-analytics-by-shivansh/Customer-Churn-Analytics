"""Unit & integration tests covering preprocessing, training, impact, and API."""

from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from churn_engine.api import app
from churn_engine.config import ChurnConfig
from churn_engine.impact import BusinessImpactAnalyzer
from churn_engine.preprocessor import ChurnDataPreprocessor, generate_sample_churn_data
from churn_engine.trainer import ChurnModelTrainer


def test_preprocessor_clean_and_engineer(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("random_state: 42\n", encoding="utf-8")
    cfg = ChurnConfig(str(cfg_path))
    prep = ChurnDataPreprocessor(cfg)

    df = generate_sample_churn_data(n_samples=40, random_state=42)
    df.loc[1] = df.loc[0]
    df.loc[2, "monthly_charges"] = None

    cleaned = prep.clean_data(df)
    assert len(cleaned) == 39
    assert not cleaned["monthly_charges"].isna().any()

    featured = prep.engineer_features(cleaned)
    for col in [
        "tenure_years",
        "is_new_customer",
        "is_loyal_customer",
        "total_revenue",
        "avg_monthly_spend",
        "is_month_to_month",
    ]:
        assert col in featured.columns

    X, y = prep.prepare_features(featured, fit=True)
    assert len(X) == len(y) == 39

    X_inf, _ = prep.prepare_features(featured.drop(columns=["churn"]).iloc[:3], fit=False)
    assert list(X_inf.columns) == list(X.columns)


def test_trainer_and_business_impact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
experiment_name: "Test_Churn"
test_size: 0.25
random_state: 42
intervention_threshold: 0.55
intervention_success_rate: 0.50
avg_customer_annual_value: 1200.0
""",
        encoding="utf-8",
    )
    cfg = ChurnConfig(str(cfg_path))
    prep = ChurnDataPreprocessor(cfg)
    df = prep.engineer_features(prep.clean_data(generate_sample_churn_data(n_samples=250)))
    X, y = prep.prepare_features(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    trainer = ChurnModelTrainer(cfg)
    metrics = trainer.train_and_evaluate(X_train, X_test, y_train, y_test, cv=2)
    assert trainer.best_model_name in {"Logistic Regression", "Random Forest"}
    assert metrics[trainer.best_model_name]["roc_auc"] > 0.65

    y_prob = trainer.best_model.predict_proba(X_test)[:, 1]
    analyzer = BusinessImpactAnalyzer(cfg)
    impact = analyzer.calculate_retention_improvement(y_test, y_prob)
    assert "annual_revenue_saved" in impact
    assert impact["new_retention_rate"] >= impact["baseline_retention_rate"]


def test_fastapi_endpoints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "churn_config.yaml").write_text(
        'experiment_name: "API_Test"\nrandom_state: 42\n', encoding="utf-8"
    )
    client = TestClient(app)

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    res_pred = client.post(
        "/predict",
        json={
            "customer_id": "CUST-101",
            "tenure_months": 3,
            "monthly_charges": 95.0,
            "contract_type": "Month-to-month",
            "support_tickets": 4,
            "payment_method": "Electronic check",
        },
    )
    assert res_pred.status_code == 200
    data = res_pred.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_tier"] in {"HIGH", "MEDIUM", "LOW"}
