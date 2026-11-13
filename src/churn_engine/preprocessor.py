"""Data loading, cleaning, feature engineering, and encoding/scaling."""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from churn_engine.config import ChurnConfig, setup_logging

logger = setup_logging()


def generate_sample_churn_data(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """Generate deterministic sample subscription/banking churn data if CSV is absent."""
    rng = np.random.default_rng(random_state)
    tenure_months = rng.integers(1, 72, size=n_samples)
    monthly_charges = np.round(rng.uniform(25.0, 120.0, size=n_samples), 2)
    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"], p=[0.55, 0.25, 0.20], size=n_samples
    )
    support_tickets = rng.poisson(lam=1.2, size=n_samples)
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_samples
    )

    logit = (
        -1.8
        - 0.04 * tenure_months
        + 0.025 * (monthly_charges - 60.0)
        + 0.55 * support_tickets
        + np.where(contract_type == "Month-to-month", 1.25, -0.75)
        + np.where(payment_method == "Electronic check", 0.45, -0.15)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = rng.binomial(1, prob)

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{10000 + i}" for i in range(n_samples)],
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "contract_type": contract_type,
            "support_tickets": support_tickets,
            "payment_method": payment_method,
            "churn": churn,
        }
    )


class ChurnDataPreprocessor:
    """End-to-end preprocessing, feature engineering, encoding, and scaling."""

    def __init__(self, config: ChurnConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_names: list[str] | None = None
        self.numeric_cols: list[str] = []

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from CSV, or bootstrap synthetic dataset if default path does not exist."""
        path = Path(data_path)
        if not path.exists():
            logger.warning(f"{data_path} not found; generating deterministic synthetic dataset.")
            path.parent.mkdir(parents=True, exist_ok=True)
            df = generate_sample_churn_data(random_state=self.config.get("random_state", 42))
            df.to_csv(path, index=False)
            return df

        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(path)
        required_cols = ["customer_id", "churn"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        logger.info(f"Loaded {len(df)} records with {df.shape[1]} columns")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate records and impute missing numeric/categorical fields."""
        logger.info("Cleaning data...")
        cleaned = df.copy()
        original_len = len(cleaned)
        if "customer_id" in cleaned.columns:
            cleaned = cleaned.drop_duplicates(subset=["customer_id"])
            if len(cleaned) < original_len:
                logger.warning(f"Removed {original_len - len(cleaned)} duplicate records")

        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned[col].isnull().any():
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())

        categorical_cols = cleaned.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if cleaned[col].isnull().any():
                cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])

        return cleaned

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create tenure, revenue, and contract indicator features."""
        logger.info("Engineering features...")
        featured = df.copy()

        if "tenure_months" in featured.columns:
            featured["tenure_years"] = featured["tenure_months"] / 12.0
            featured["is_new_customer"] = (featured["tenure_months"] <= 6).astype(int)
            featured["is_loyal_customer"] = (featured["tenure_months"] >= 24).astype(int)

        if "monthly_charges" in featured.columns and "tenure_months" in featured.columns:
            featured["total_revenue"] = featured["monthly_charges"] * featured["tenure_months"]
            featured["avg_monthly_spend"] = featured["total_revenue"] / (
                featured["tenure_months"] + 1.0
            )

        if "contract_type" in featured.columns:
            featured["is_month_to_month"] = (
                featured["contract_type"].str.lower() == "month-to-month"
            ).astype(int)

        return featured

    def prepare_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series | None]:
        """Encode categoricals and scale numeric columns for training or inference."""
        logger.info("Preparing features...")
        y = df["churn"] if "churn" in df.columns else None
        X = df.drop(["churn", "customer_id"], axis=1, errors="ignore").copy()

        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            elif col in self.label_encoders:
                known = set(self.label_encoders[col].classes_)
                fallback = self.label_encoders[col].classes_[0]
                mapped = X[col].astype(str).apply(lambda v: v if v in known else fallback)
                X[col] = self.label_encoders[col].transform(mapped)
            else:
                X = X.drop(col, axis=1)

        if fit:
            self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            X[self.numeric_cols] = self.scaler.fit_transform(X[self.numeric_cols])
            self.feature_names = X.columns.tolist()
        else:
            X = X.reindex(columns=self.feature_names, fill_value=0.0)
            X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])

        return X, y
