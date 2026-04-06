"""
Axiom MLOps Project: Funding Prediction Engine.

This module implements a professional machine learning pipeline using the
Strategy Pattern for model selection and Weights & Biases for experiment tracking.
"""

import sys
from pathlib import Path
from typing import List, Literal, Optional, Protocol, Tuple, Union, runtime_checkable

import joblib
import pandas as pd
import wandb
from loguru import logger
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# --- Configuration Patterns ---


class RandomForestConfig(BaseModel):
    """Configuration schema for Random Forest Classifier."""

    KIND: Literal["RandomForest"] = "RandomForest"
    n_estimators: int = Field(default=100, gt=0, description="Number of trees in forest")
    max_depth: int = Field(default=10, gt=0, description="Maximum depth of trees")
    random_state: int = 42


class SVMConfig(BaseModel):
    """Configuration schema for Support Vector Machine."""

    KIND: Literal["SVM"] = "SVM"
    C: float = Field(default=1.0, gt=0, description="Regularization parameter")
    kernel: str = Field(default="rbf", description="Kernel type (rbf, poly, linear)")
    probability: bool = True


ModelConfig = Union[RandomForestConfig, SVMConfig]


@runtime_checkable
class ModelStrategy(Protocol):
    """
    Protocol defining the interface for all model strategies.

    Ensures compatibility across different sklearn estimators within
    the Strategy Pattern.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict_proba(self, X: pd.DataFrame) -> List[List[float]]: ...


# --- The Axiom Engine ---


class FundingPredictor:
    """
    MLOps engine for predicting funding success.

    Attributes:
        config (ModelConfig): Validated configuration for the chosen model.
        base_dir (Path): Resolved project root directory.
        data_path (Path): Path to the input CSV data.
        model_path (Path): Path to the serialized model file.
    """

    def __init__(self, config: ModelConfig = RandomForestConfig()) -> None:
        """
        Initializes the predictor with directory resolution and logging.

        Args:
            config: Configuration object (RandomForest or SVM).
        """
        self.base_dir = Path(__file__).resolve().parents[2]
        if Path("/app/data").exists():
            self.base_dir = Path("/app")

        self.data_path = self.base_dir / "data" / "funding_data.csv"
        self.model_path = self.base_dir / "models" / "funding_model.joblib"
        self.features_path = self.base_dir / "models" / "model_features.joblib"

        self.config = config
        self.model: Optional[ModelStrategy] = None
        self.feature_names: List[str] = []

        logger.remove()
        logger.add(sys.stderr, level="INFO")

    def _model_factory(self) -> ModelStrategy:
        """
        Factory method to instantiate the model based on configuration.

        Returns:
            An instantiated sklearn classifier matching ModelStrategy.
        """
        if isinstance(self.config, RandomForestConfig):
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
            )
        elif isinstance(self.config, SVMConfig):
            return SVC(C=self.config.C, kernel=self.config.kernel, probability=True)
        raise NotImplementedError(f"Strategy {self.config.KIND} is not implemented.")

    def load_and_clean_data(self) -> pd.DataFrame:
        """
        Loads raw data and prepares it for training.

        Returns:
            pd.DataFrame: Cleaned data containing only valid target statuses.
        """
        df = pd.read_csv(self.data_path)
        df_clean = df[df["fund_order_status"].isin(["SUCCESS", "FAILED"])].copy()
        df_clean["target"] = (df_clean["fund_order_status"] == "SUCCESS").astype(int)
        return df_clean

    def train(self) -> None:
        """
        Executes the full training pipeline and logs results to Weights & Biases.

        Steps:
            1. Initialize W&B run.
            2. Load/Clean data.
            3. Feature engineering (one-hot encoding).
            4. Model fitting.
            5. Artifact serialization and W&B logging.
        """
        wandb.init(
            project="axiom-mlops-project",
            config=self.config.model_dump(),
            name=f"train-{self.config.KIND}",
        )

        logger.info("🚀 Training Pipeline Start [Strategy: {}]", self.config.KIND)
        df = self.load_and_clean_data()

        X = pd.get_dummies(df[["fund_type", "fund_amount"]], drop_first=True)
        y = df["target"]
        self.feature_names = X.columns.tolist()

        self.model = self._model_factory()
        self.model.fit(X, y)

        wandb.log({"feature_count": len(self.feature_names)})

        self.model_path.parent.mkdir(exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.feature_names, self.features_path)

        wandb.save(str(self.model_path))

        logger.success("Model trained successfully and tracked in W&B.")
        wandb.finish()

    def predict(self, fund_type: str, amount: float) -> Tuple[str, float]:
        """
        Performs inference on a single funding request.

        Args:
            fund_type: The type of funding (e.g., 'TRANSFER_P2P').
            amount: The monetary value of the request.

        Returns:
            Tuple[str, float]: (Result string 'SUCCESS'/'FAILED', Confidence probability).
        """
        if self.model is None:
            self.model = joblib.load(self.model_path)
            self.feature_names = joblib.load(self.features_path)

        # Prepare input vector
        input_df = pd.DataFrame(0, index=[0], columns=self.feature_names)
        input_df["fund_amount"] = amount
        col_name = f"fund_type_{fund_type}"

        if col_name in input_df.columns:
            input_df[col_name] = 1

        proba = float(self.model.predict_proba(input_df)[0][1])
        result = "SUCCESS" if proba > 0.5 else "FAILED"

        logger.info("Inference: {} (Conf: {:.2%})", result, proba)
        return result, proba


def main() -> None:
    """Entry point for the Axiom Engine training and inference CLI."""
    config = RandomForestConfig(n_estimators=200, max_depth=8)
    predictor = FundingPredictor(config=config)
    predictor.train()
    predictor.predict("TRANSFER_P2P", 1250.0)


if __name__ == "__main__":
    main()
