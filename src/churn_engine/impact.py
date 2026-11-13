"""Retention campaign ROI and business impact simulator."""

from pathlib import Path
from typing import Dict
import mlflow
import numpy as np
import pandas as pd

from churn_engine.config import ChurnConfig, setup_logging

logger = setup_logging()


class BusinessImpactAnalyzer:
    """Calculates projected retention lift and revenue saved from targeted interventions."""

    def __init__(self, config: ChurnConfig):
        self.config = config
        Path("reports").mkdir(parents=True, exist_ok=True)

    def calculate_retention_improvement(
        self, y_test: pd.Series, y_pred_proba: np.ndarray, threshold: float | None = None
    ) -> Dict:
        thresh = (
            threshold
            if threshold is not None
            else float(self.config.get("intervention_threshold", 0.7))
        )
        baseline_retention = float(1.0 - np.mean(y_test))
        high_risk_mask = y_pred_proba > thresh
        high_risk_count = int(np.sum(high_risk_mask))

        success_rate = float(self.config.get("intervention_success_rate", 0.50))
        saved_customers = int(high_risk_count * success_rate)

        total_customers = len(y_test)
        baseline_churners = int(np.sum(y_test))
        new_churners = max(baseline_churners - saved_customers, 0)
        new_retention = 1.0 - (new_churners / total_customers)

        retention_improvement_pct = (
            ((new_retention - baseline_retention) / baseline_retention) * 100.0
            if baseline_retention > 0
            else 0.0
        )
        avg_customer_value = float(self.config.get("avg_customer_annual_value", 1200.0))
        revenue_saved = saved_customers * avg_customer_value

        impact = {
            "baseline_retention_rate": round(baseline_retention, 4),
            "new_retention_rate": round(new_retention, 4),
            "retention_improvement_pct": round(retention_improvement_pct, 2),
            "customers_targeted": high_risk_count,
            "customers_saved": saved_customers,
            "annual_revenue_saved": round(revenue_saved, 2),
        }

        if mlflow.active_run():
            for key, value in impact.items():
                mlflow.log_metric(key, value)

        pd.DataFrame([impact]).to_csv("reports/business_impact.csv", index=False)
        return impact
