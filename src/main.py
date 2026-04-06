from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class FundingPredictor:
    """
    A production-grade class to manage the funding success prediction pipeline.

    This class handles data loading, model training, and inference using a
    Random Forest architecture. It follows the Hybrid OOP/Functional paradigm.
    """

    def __init__(self):
        """Initializes the predictor with project-standard paths."""
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_path = self.base_dir / "data" / "funding_data.csv"
        self.model_path = self.base_dir / "models" / "funding_model_rf.joblib"
        self.features_path = self.base_dir / "models" / "model_features.joblib"
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: List[str] = []

    def load_and_clean_data(self) -> pd.DataFrame:
        """
        Loads raw CSV data and filters for terminal order statuses.

        Returns:
            pd.DataFrame: A cleaned dataframe containing only SUCCESS/FAILED rows.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"⚠️ Data not found at {self.data_path}")

        df = pd.read_csv(self.data_path)
        valid_statuses = ["SUCCESS", "FAILED"]
        df_clean = df[df["fund_order_status"].isin(valid_statuses)].copy()

        # Explicit target mapping
        df_clean["target"] = 0
        df_clean.loc[df_clean["fund_order_status"] == "SUCCESS", "target"] = 1

        return df_clean

    def train(self) -> None:
        """
        Orchestrates the training pipeline: Clean -> Encode -> Train -> Save.
        """
        print("🚀 Starting Production Training Pipeline...")
        df = self.load_and_clean_data()

        # Feature Engineering
        X = pd.get_dummies(df[["fund_type", "fund_amount"]], drop_first=True)
        y = df["target"]
        self.feature_names = X.columns.tolist()

        # Model Training
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
        )
        self.model.fit(X, y)

        # Persistence (Encapsulation)
        self.model_path.parent.mkdir(exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.feature_names, self.features_path)
        print(f"✅ Model and Features saved to {self.model_path.parent}")

    def _load_model_assets(self):
        """Internal helper to load model from disk if not in memory."""
        if self.model is None:
            self.model = joblib.load(self.model_path)
            self.feature_names = joblib.load(self.features_path)

    def predict(self, fund_type: str, amount: float) -> Tuple[str, float]:
        """
        Predicts the success of a funding event based on type and amount.

        Args:
            fund_type (str): The category of funding (e.g., 'TRANSFER_P2P').
            amount (float): The numerical value of the transaction.

        Returns:
            Tuple[str, float]: A tuple containing the (Result String, Probability).
        """
        self._load_model_assets()

        # Prepare input vector
        input_df = pd.DataFrame(0, index=[0], columns=self.feature_names)
        input_df["fund_amount"] = amount

        col_name = f"fund_type_{fund_type}"
        if col_name in input_df.columns:
            input_df[col_name] = 1

        proba = self.model.predict_proba(input_df)[0][1]
        result = "SUCCESS" if proba > 0.5 else "FAILED"

        return result, proba


def main():
    """Main Entrypoint for the application."""
    predictor = FundingPredictor()
    print("--- 🏛️ Axiom MLOps OOP Engine ---")

    if not predictor.model_path.exists():
        predictor.train()
    else:
        print("🟢 Model detected. Engine Online.")
        # Quick health check test
        res, conf = predictor.predict("TRANSFER_P2P", 1000)
        print(f"Health Check: {res} ({conf:.2%})")


if __name__ == "__main__":
    main()
