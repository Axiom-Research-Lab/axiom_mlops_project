"""
Axiom MLOps Project: Funding Prediction Engine.

This module implements a professional machine learning pipeline using the
Strategy Pattern for model selection and Weights & Biases for Model Registry management.
"""

import sys
from pathlib import Path
from typing import List, Literal, Optional, Protocol, Tuple, Union, runtime_checkable

import joblib
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import wandb

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
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict_proba(self, X: pd.DataFrame) -> List[List[float]]: ...


# --- The Axiom Engine ---


class FundingPredictor:
    """
    MLOps engine for predicting funding success with Registry Support.
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

    def load_and_clean_data(self) -> pd.DataFrame:
        """Loads and prepares raw data."""
        df = pd.read_csv(self.data_path)
        df_clean = df[df["fund_order_status"].isin(["SUCCESS", "FAILED"])].copy()
        df_clean["target"] = (df_clean["fund_order_status"] == "SUCCESS").astype(int)
        return df_clean

    def train(self) -> None:
        """
        Trains and registers the model in the W&B Model Registry.
        """
        run = wandb.init(
            project="axiom-mlops-project",
            config=self.config.model_dump(),
            name=f"train-{self.config.KIND}",
            job_type="train",
        )

        logger.info("🚀 Training Pipeline Start [Strategy: {}]", self.config.KIND)
        df = self.load_and_clean_data()

        X = pd.get_dummies(df[["fund_type", "fund_amount"]], drop_first=True)
        y = df["target"]
        self.feature_names = X.columns.tolist()

        self.model = self._model_factory()
        self.model.fit(X, y)

        # 1. Save locally
        self.model_path.parent.mkdir(exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.feature_names, self.features_path)

        # 2. Register as an Artifact (The Model Registry part)
        model_artifact = wandb.Artifact(
            name="funding-model",
            type="model",
            description=f"Axiom core {self.config.KIND} model",
            metadata={**self.config.model_dump(), "feature_names": self.feature_names},
        )
        model_artifact.add_file(str(self.model_path))
        model_artifact.add_file(str(self.features_path))

        # 3. Log to registry
        run.log_artifact(model_artifact)

        logger.success("Model trained and registered to W&B successfully.")
        wandb.finish()

    def load_champion(self, alias: str = "champion") -> None:
        """
        Downloads and loads the model version tagged with the given alias.
        Fulfills the 'Loading from Registry' requirement of Section 5.6.
        """
        run = wandb.init(project="axiom-mlops-project", job_type="inference")

        logger.info("Fetching model with alias: '{}' from registry...", alias)
        artifact = run.use_artifact(f"funding-model:{alias}", type="model")
        artifact_dir = Path(artifact.download())

        self.model = joblib.load(artifact_dir / "funding_model.joblib")
        self.feature_names = joblib.load(artifact_dir / "model_features.joblib")

        logger.success("Champion model loaded from registry.")
        run.finish()

    def predict(self, fund_type: str, amount: float) -> Tuple[str, float]:
        """Performs inference on a single funding request."""
        if self.model is None:
            logger.warning("No model in memory. Loading local fallback...")
            self.model = joblib.load(self.model_path)
            self.feature_names = joblib.load(self.features_path)

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
    """Entry point for training and registry testing."""
    # 1. Train and Register a new version
    config = RandomForestConfig(n_estimators=150, max_depth=6)
    predictor = FundingPredictor(config=config)
    predictor.train()

    # 2. Demonstrate Registry Loading
    # Note: This will fail until you manually tag a version as 'champion' in W&B UI
    # or via the API script I gave you earlier.
    try:
        predictor.load_champion(alias="latest")  # Using 'latest' as a safe default
        predictor.predict("TRANSFER_P2P", 1500.0)
    except Exception as e:
        logger.error("Could not load from registry: {}", e)


if __name__ == "__main__":
    main()
