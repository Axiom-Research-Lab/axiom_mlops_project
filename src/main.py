import sys
from pathlib import Path
from typing import List, Optional, Tuple

import joblib  # type: ignore
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier  # type: ignore

# --- 4.3 Logging Configuration ---
# Remove default logger and add custom sinks
logger.remove()
# Sink 1: Terminal (with colors)
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
# Sink 2: File (with rotation)
logger.add(
    "axiom_engine.log",
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    level="DEBUG",
)


class FundingPredictor:
    """
    A production-grade class to manage the funding success prediction pipeline.
    """

    def __init__(self) -> None:
        """Initializes the predictor with project-standard paths."""
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_path = self.base_dir / "data" / "funding_data.csv"
        self.model_path = self.base_dir / "models" / "funding_model_rf.joblib"
        self.features_path = self.base_dir / "models" / "model_features.joblib"
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: List[str] = []
        logger.debug("FundingPredictor initialized with base_dir: {}", self.base_dir)

    def load_and_clean_data(self) -> pd.DataFrame:
        """Loads raw CSV data and filters for terminal order statuses."""
        if not self.data_path.exists():
            logger.error("Data file not found at path: {}", self.data_path)
            raise FileNotFoundError(f"⚠️ Data not found at {self.data_path}")

        logger.info("Loading dataset from {}...", self.data_path.name)
        df = pd.read_csv(self.data_path)
        valid_statuses = ["SUCCESS", "FAILED"]
        df_clean = df[df["fund_order_status"].isin(valid_statuses)].copy()

        df_clean["target"] = 0
        df_clean.loc[df_clean["fund_order_status"] == "SUCCESS", "target"] = 1

        logger.debug("Cleaned data shape: {}", df_clean.shape)
        return df_clean

    def train(self) -> None:
        """Orchestrates the training pipeline: Clean -> Encode -> Train -> Save."""
        logger.info("🚀 Starting Production Training Pipeline...")
        df = self.load_and_clean_data()

        X = pd.get_dummies(df[["fund_type", "fund_amount"]], drop_first=True)
        y = df["target"]
        self.feature_names = X.columns.tolist()

        logger.info("Training Random Forest with {} features...", len(self.feature_names))
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
        )
        self.model.fit(X, y)

        self.model_path.parent.mkdir(exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.feature_names, self.features_path)
        logger.success("Model and Features saved successfully to {}", self.model_path.parent)

    def _load_model_assets(self) -> None:
        """Internal helper to load model assets from disk."""
        if self.model is None:
            if not self.model_path.exists():
                logger.warning(
                    "Attempted to load model, but file does not exist at {}",
                    self.model_path,
                )
                return
            self.model = joblib.load(self.model_path)
            self.feature_names = joblib.load(self.features_path)
            logger.debug("Model assets loaded into memory.")

    def predict(self, fund_type: str, amount: float) -> Tuple[str, float]:
        """Predicts the success of a funding event."""
        self._load_model_assets()

        if self.model is None:
            logger.critical("Prediction failed: Model not initialized.")
            raise RuntimeError("Axiom Engine Error: Model assets are not initialized.")

        input_df = pd.DataFrame(0, index=[0], columns=self.feature_names)
        input_df["fund_amount"] = amount
        col_name = f"fund_type_{fund_type}"

        if col_name in input_df.columns:
            input_df[col_name] = 1
        else:
            logger.warning("Unknown fund_type encountered: {}", fund_type)

        proba = float(self.model.predict_proba(input_df)[0][1])
        result = "SUCCESS" if proba > 0.5 else "FAILED"

        logger.info("Prediction generated for {}: {} ({:.2%})", fund_type, result, proba)
        return result, proba


def main() -> None:
    """Main Entrypoint."""
    predictor = FundingPredictor()
    logger.info("--- 🏛️ Axiom MLOps OOP Engine ---")

    if not predictor.model_path.exists():
        predictor.train()
    else:
        logger.info("🟢 Model detected. Engine Online.")
        res, conf = predictor.predict("TRANSFER_P2P", 1000.0)
        logger.success("Health Check Completed: {} ({:.2%})", res, conf)


if __name__ == "__main__":
    main()
