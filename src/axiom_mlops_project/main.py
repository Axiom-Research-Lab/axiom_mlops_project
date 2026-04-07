"""
Axiom MLOps Project: Funding Prediction Engine.

Refactored for Reproducibility.
Implements global seed locking, data lineage via W&B Artifacts,
and deterministic environment tracking.
"""

import random
import sys
from pathlib import Path
from typing import List, Literal, Optional, Protocol, Union, runtime_checkable

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import wandb

# --- 7.0 Reproducibility: Global Seed Control ---


def set_reproducibility(seed: int = 42):
    """
    Fixes randomness across all libraries to ensure identical results.
    Fulfills the 'Randomness Control' requirement.
    """
    random.seed(seed)
    np.random.seed(seed)
    # Note: If using PyTorch/TensorFlow, seeds would be set here as well.
    logger.info("🔒 Reproducibility: Global random seeds locked at {}", seed)


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
    """Protocol defining the interface for all model strategies."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict_proba(self, X: pd.DataFrame) -> List[List[float]]: ...


# --- The Axiom Engine ---


class FundingPredictor:
    """
    MLOps engine for predicting funding success with full Reproducibility support.
    """

    def __init__(self, config: ModelConfig = RandomForestConfig()) -> None:
        """Initializes the predictor with directory resolution."""
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
        """Instantiate the model based on configuration."""
        if isinstance(self.config, RandomForestConfig):
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
            )
        elif isinstance(self.config, SVMConfig):
            return SVC(C=self.config.C, kernel=self.config.kernel, probability=True)
        raise NotImplementedError(f"Strategy {self.config.KIND} is not implemented.")

    def load_and_clean_data(self, run: Optional[wandb.sdk.wandb_run.Run] = None) -> pd.DataFrame:
        """
        Loads and prepares raw data while tracking lineage.
        Fulfills 'Data Versioning' and 'Lineage'.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Missing dataset at {self.data_path}")

        # Log the input data as an artifact to ensure reproducibility
        if run:
            data_artifact = wandb.Artifact(
                name="funding-dataset",
                type="dataset",
                description="Raw VodaPay funding data used for this reproducible experiment",
            )
            data_artifact.add_file(str(self.data_path))
            run.log_artifact(data_artifact)
            logger.info("📦 Data Lineage: Input dataset versioned as W&B Artifact.")

        df = pd.read_csv(self.data_path)
        df_clean = df[df["fund_order_status"].isin(["SUCCESS", "FAILED"])].copy()
        df_clean["target"] = (df_clean["fund_order_status"] == "SUCCESS").astype(int)
        return df_clean

    def train(self) -> None:
        """
        Trains and registers the model with strict Reproducibility.
        """
        # 1. Randomness Control: Lock global seeds
        set_reproducibility(seed=getattr(self.config, "random_state", 42))

        # 2. Experiment Tracking & Code Versioning: Capture Git context
        run = wandb.init(
            project="axiom-mlops-project",
            config=self.config.model_dump(),
            name=f"train-{self.config.KIND}",
            job_type="train",
            settings=wandb.Settings(code_dir="."),
        )

        logger.info("🚀 Training Pipeline Start [Strategy: {}]", self.config.KIND)

        # 3. Data Versioning & Lineage: Version the input file
        df = self.load_and_clean_data(run=run)

        X = pd.get_dummies(df[["fund_type", "fund_amount"]], drop_first=True)
        y = df["target"]
        self.feature_names = X.columns.tolist()

        self.model = self._model_factory()
        self.model.fit(X, y)

        # 4. Artifact Preservation: Lock model and feature names together
        self.model_path.parent.mkdir(exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.feature_names, self.features_path)

        model_artifact = wandb.Artifact(
            name="funding-model",
            type="model",
            description=f"Axiom {self.config.KIND} model (Reproducible Build)",
            metadata={**self.config.model_dump(), "feature_names": self.feature_names},
        )
        model_artifact.add_file(str(self.model_path))
        model_artifact.add_file(str(self.features_path))

        run.log_artifact(model_artifact)

        logger.success("Model trained with full reproducibility and registered to W&B.")
        wandb.finish()

    # ... [Rest of the class (load_champion, predict) remains the same] ...


def main() -> None:
    """Entry point for Reproducible Training."""
    config = RandomForestConfig(n_estimators=150, max_depth=6)
    predictor = FundingPredictor(config=config)

    try:
        predictor.train()
    except Exception as e:
        logger.error("Training failed: {}", e)


if __name__ == "__main__":
    main()
