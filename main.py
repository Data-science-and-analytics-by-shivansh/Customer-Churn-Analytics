import numpy as np
import pandas as pd
import mlflow
import logging
from datetime import datetime
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
from imblearn.over_sampling import SMOTE
import joblib
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# CONFIGURATION
# =====================================================
class ChurnConfig:
    def __init__(self, config_path="churn_config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
    
    def get(self, key, default=None):
        return self.cfg.get(key, default)

# =====================================================
# LOGGING
# =====================================================
def setup_logging(log_level="INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/churn_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =====================================================
# DATA PREPROCESSING
# =====================================================
class ChurnDataPreprocessor:
    def __init__(self, config: ChurnConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data with validation"""
        try:
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            
            # Validate required columns
            required_cols = ['customer_id', 'churn']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            logger.info(f"Loaded {len(df)} records with {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        logger.info("Cleaning data...")
        
        # Remove duplicates
        original_len = len(df)
        df = df.drop_duplicates(subset=['customer_id'])
        if len(df) < original_len:
            logger.warning(f"Removed {original_len - len(df)} duplicate records")
        
        # Handle missing values
        missing_summary = df.isnull().sum()
        if missing_summary.any():
            logger.info(f"Missing values:\n{missing_summary[missing_summary > 0]}")
            
            # Fill numeric with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
            
            # Fill categorical with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features"""
        logger.info("Engineering features...")
        
        # Tenure-based features
        if 'tenure_months' in df.columns:
            df['tenure_years'] = df['tenure_months'] / 12
            df['is_new_customer'] = (df['tenure_months'] <= 6).astype(int)
            df['is_loyal_customer'] = (df['tenure_months'] >= 24).astype(int)
        
        # Revenue-based features
        if 'monthly_charges' in df.columns and 'tenure_months' in df.columns:
            df['total_revenue'] = df['monthly_charges'] * df['tenure_months']
            df['avg_monthly_spend'] = df['total_revenue'] / (df['tenure_months'] + 1)
        
        # Contract-based features
        if 'contract_type' in df.columns:
            df['is_month_to_month'] = (df['contract_type'] == 'Month-to-month').astype(int)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for modeling"""
        logger.info("Preparing features...")
        
        # Separate target
        if 'churn' not in df.columns:
            raise ValueError("Target column 'churn' not found")
        
        y = df['churn']
        X = df.drop(['churn', 'customer_id'], axis=1, errors='ignore')
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
                else:
                    logger.warning(f"No encoder found for {col}, dropping column")
                    X = X.drop(col, axis=1)
        
        # Scale numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit:
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
            self.feature_names = X.columns.tolist()
        else:
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        
        logger.info(f"Features prepared: {X.shape[1]} features")
        return X, y

# =====================================================
# MODEL TRAINING & EVALUATION
# =====================================================
class ChurnModelTrainer:
    def __init__(self, config: ChurnConfig):
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}
        
    def handle_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance with SMOTE"""
        logger.info("Handling class imbalance with SMOTE...")
        
        # Check class distribution
        class_dist = y_train.value_counts()
        logger.info(f"Original distribution:\n{class_dist}")
        
        imbalance_ratio = class_dist.min() / class_dist.max()
        
        if imbalance_ratio < 0.5:  # If minority class < 50% of majority
            smote = SMOTE(random_state=42, sampling_strategy='auto')
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            logger.info(f"After SMOTE:\n{pd.Series(y_train_balanced).value_counts()}")
            return X_train_balanced, y_train_balanced
        else:
            logger.info("Classes reasonably balanced, skipping SMOTE")
            return X_train, y_train
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train Logistic Regression with hyperparameter tuning"""
        logger.info("Training Logistic Regression...")
        
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'class_weight': ['balanced', None]
        }
        
        lr = LogisticRegression(random_state=42, max_iter=1000)
        
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best LR params: {grid_search.best_params_}")
        logger.info(f"Best LR CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        self.models['Logistic Regression'] = grid_search.best_estimator_
        
        # Log to MLflow
        mlflow.log_params({f"lr_{k}": v for k, v in grid_search.best_params_.items()})
        mlflow.log_metric("lr_cv_roc_auc", grid_search.best_score_)
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train Random Forest with hyperparameter tuning"""
        logger.info("Training Random Forest...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced', None]
        }
        
        rf = RandomForestClassifier(random_state=42)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best RF params: {grid_search.best_params_}")
        logger.info(f"Best RF CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        self.models['Random Forest'] = grid_search.best_estimator_
        
        # Log to MLflow
        mlflow.log_params({f"rf_{k}": v for k, v in grid_search.best_params_.items()})
        mlflow.log_metric("rf_cv_roc_auc", grid_search.best_score_)
    
    def evaluate_model(self, model_name: str, model, X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict:
        """Comprehensive model evaluation"""
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Churn class metrics (class 1)
        metrics['churn_precision'] = report['1']['precision']
        metrics['churn_recall'] = report['1']['recall']
        metrics['churn_f1'] = report['1']['f1-score']
        
        # Log to MLflow
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"{model_name.lower().replace(' ', '_')}_{metric_name}", value)
        
        logger.info(f"{model_name} Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  Churn Recall: {metrics['churn_recall']:.4f}")
        logger.info(f"  Churn Precision: {metrics['churn_precision']:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, model_name)
        
        # ROC Curve
        self._plot_roc_curve(y_test, y_pred_proba, model_name)
        
        return metrics
    
    def _plot_confusion_matrix(self, cm, model_name: str):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Churn', 'Churn'],
                   yticklabels=['Not Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        filepath = f"visualizations/{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        mlflow.log_artifact(filepath)
    
    def _plot_roc_curve(self, y_test, y_pred_proba, model_name: str):
        """Plot and save ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(alpha=0.3)
        
        filepath = f"visualizations/{model_name.lower().replace(' ', '_')}_roc_curve.png"
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        mlflow.log_artifact(filepath)
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Complete training and evaluation pipeline"""
        mlflow.set_experiment(self.config.get('experiment_name', 'Churn_Prediction'))
        
        with mlflow.start_run(run_name="Churn_Model_Training"):
            # Log dataset info
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("churn_rate_train", y_train.mean())
            mlflow.log_param("churn_rate_test", y_test.mean())
            
            # Handle imbalance
            X_train_balanced, y_train_balanced = self.handle_imbalance(X_train, y_train)
            
            # Train models
            self.train_logistic_regression(X_train_balanced, y_train_balanced)
            self.train_random_forest(X_train_balanced, y_train_balanced)
            
            # Evaluate models
            for model_name, model in self.models.items():
                metrics = self.evaluate_model(model_name, model, X_test, y_test)
                self.metrics[model_name] = metrics
            
            # Select best model based on ROC-AUC
            best_model_name = max(self.metrics, key=lambda k: self.metrics[k]['roc_auc'])
            self.best_model_name = best_model_name
            self.best_model = self.models[best_model_name]
            
            logger.info(f"\n=== BEST MODEL: {best_model_name} ===")
            logger.info(f"ROC-AUC: {self.metrics[best_model_name]['roc_auc']:.4f}")
            
            mlflow.log_param("best_model", best_model_name)
            
            # Feature importance (for Random Forest)
            if best_model_name == 'Random Forest':
                self._analyze_feature_importance(X_train)
            elif best_model_name == 'Logistic Regression':
                self._analyze_coefficients(X_train)
            
            # Save model
            self._save_model()
    
    def _analyze_feature_importance(self, X_train):
        """Analyze and visualize feature importance"""
        logger.info("Analyzing feature importance...")
        
        importances = self.best_model.feature_importances_
        feature_names = X_train.columns
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 10 Features:\n{importance_df.head(10)}")
        
        # Plot
        plt.figure(figsize=(10, 8))
        top_n = 15
        top_features = importance_df.head(top_n)
        plt.barh(range(top_n), top_features['importance'])
        plt.yticks(range(top_n), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances - Random Forest')
        plt.gca().invert_yaxis()
        
        filepath = "visualizations/feature_importance.png"
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        mlflow.log_artifact(filepath)
        
        # Save CSV
        importance_df.to_csv("reports/feature_importance.csv", index=False)
        mlflow.log_artifact("reports/feature_importance.csv")
    
    def _analyze_coefficients(self, X_train):
        """Analyze logistic regression coefficients"""
        logger.info("Analyzing model coefficients...")
        
        coefficients = self.best_model.coef_[0]
        feature_names = X_train.columns
        
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        logger.info(f"Top 10 Features by Coefficient:\n{coef_df.head(10)}")
        
        # Save CSV
        coef_df.to_csv("reports/model_coefficients.csv", index=False)
        mlflow.log_artifact("reports/model_coefficients.csv")
    
    def _save_model(self):
        """Save trained model and preprocessor"""
        model_path = "models/churn_model.pkl"
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'metrics': self.metrics[self.best_model_name]
        }
        
        joblib.dump(model_data, model_path)
        mlflow.sklearn.log_model(self.best_model, "model")
        
        logger.info(f"Model saved to {model_path}")

# =====================================================
# BUSINESS IMPACT ANALYSIS
# =====================================================
class BusinessImpactAnalyzer:
    def __init__(self, config: ChurnConfig):
        self.config = config
    
    def calculate_retention_improvement(self, y_test, y_pred_proba, 
                                       threshold=0.7) -> Dict:
        """
        Calculate projected retention improvement from targeted interventions.
        
        Assumptions:
        - Baseline: No targeting, natural retention rate
        - With model: Target high-risk customers (predicted prob > threshold)
        - Intervention success rate: 50% (configurable)
        """
        logger.info("Calculating business impact...")
        
        # Baseline retention (no intervention)
        baseline_retention = 1 - y_test.mean()
        
        # Identify high-risk customers
        high_risk_mask = y_pred_proba > threshold
        high_risk_count = high_risk_mask.sum()
        
        # Assume intervention saves 50% of high-risk customers
        intervention_success_rate = self.config.get('intervention_success_rate', 0.50)
        saved_customers = int(high_risk_count * intervention_success_rate)
        
        # New retention rate
        total_customers = len(y_test)
        baseline_churners = int(y_test.sum())
        new_churners = baseline_churners - saved_customers
        new_retention = 1 - (new_churners / total_customers)
        
        # Calculate improvement
        retention_improvement_pct = ((new_retention - baseline_retention) / baseline_retention) * 100
        
        # Calculate revenue impact
        avg_customer_value = self.config.get('avg_customer_annual_value', 1200)
        revenue_saved = saved_customers * avg_customer_value
        
        impact = {
            'baseline_retention_rate': baseline_retention,
            'new_retention_rate': new_retention,
            'retention_improvement_pct': retention_improvement_pct,
            'customers_targeted': high_risk_count,
            'customers_saved': saved_customers,
            'annual_revenue_saved': revenue_saved
        }
        
        logger.info("\n=== BUSINESS IMPACT ===")
        logger.info(f"Baseline Retention: {baseline_retention:.2%}")
        logger.info(f"Projected Retention: {new_retention:.2%}")
        logger.info(f"Improvement: {retention_improvement_pct:.1f}%")
        logger.info(f"Customers Saved: {saved_customers}")
        logger.info(f"Revenue Saved: ${revenue_saved:,.0f}")
        
        # Log to MLflow
        for key, value in impact.items():
            mlflow.log_metric(key, value)
        
        # Save report
        impact_df = pd.DataFrame([impact])
        impact_df.to_csv("reports/business_impact.csv", index=False)
        mlflow.log_artifact("reports/business_impact.csv")
        
        return impact

# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":
    # Load configuration
    config = ChurnConfig()
    
    # Initialize components
    preprocessor = ChurnDataPreprocessor(config)
    trainer = ChurnModelTrainer(config)
    impact_analyzer = BusinessImpactAnalyzer(config)
    
    # Load and clean data
    data_path = config.get('data_path', 'churn_data.csv')
    df = preprocessor.load_data(data_path)
    df = preprocessor.clean_data(df)
    df = preprocessor.engineer_features(df)
    
    # Prepare features
    X, y = preprocessor.prepare_features(df, fit=True)
    
    # Train-test split (stratified to maintain class balance)
    test_size = config.get('test_size', 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train and evaluate models
    trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Calculate business impact
    y_pred_proba = trainer.best_model.predict_proba(X_test)[:, 1]
    impact = impact_analyzer.calculate_retention_improvement(y_test, y_pred_proba)
    
    # Save preprocessor
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    logger.info("\n=== TRAINING COMPLETE ===")
