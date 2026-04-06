import sys
from pathlib import Path
from typing import List, Literal, Optional, Protocol, Tuple, Union, runtime_checkable

import joblib  # type: ignore
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.svm import SVC  # type: ignore

# --- 5.0 Configuration Patterns (Pydantic) ---


class RandomForestConfig(BaseModel):
    """Configuration for Random Forest Strategy."""

    KIND: Literal["RandomForest"] = "RandomForest"
    n_estimators: int = Field(default=100, gt=0)
    max_depth: int = Field(default=10, gt=0)
    random_state: int = 42


class SVMConfig(BaseModel):
    """Configuration for SVM Strategy."""

    KIND: Literal["SVM"] = "SVM"
    C: float = Field(default=1.0, gt=0)
    kernel: str = "rbf"
    probability: bool = True


# Discriminated Union for the Factory
ModelConfig = Union[RandomForestConfig, SVMConfig]

# --- 5.0 Interface Patterns (Protocols) ---


@runtime_checkable
class ModelStrategy(Protocol):
    """The 'Contract' for our Strategy Pattern."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict_proba(self, X: pd.DataFrame) -> List[List[float]]: ...


# --- The Axiom Engine ---


class FundingPredictor:
    """Refined MLOps engine using Design Patterns."""

    def __init__(self, config: ModelConfig = RandomForestConfig()) -> None:
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_path = self.base_dir / "data" / "funding_data.csv"
        self.model_path = self.base_dir / "models" / "funding_model.joblib"
        self.features_path = self.base_dir / "models" / "model_features.joblib"

        self.config = config
        self.model: Optional[ModelStrategy] = None
        self.feature_names: List[str] = []

        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add("axiom_engine.log", rotation="10 MB", level="DEBUG")

    def _model_factory(self) -> ModelStrategy:
        """Factory Pattern implementation."""
        if isinstance(self.config, RandomForestConfig):
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
            )
        elif isinstance(self.config, SVMConfig):
            return SVC(
                C=self.config.C, kernel=self.config.kernel, probability=self.config.probability
            )
        raise NotImplementedError(f"Strategy {self.config.KIND} not supported.")

    def load_and_clean_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            logger.error("Data source missing at {}", self.data_path)
            raise FileNotFoundError(f"Missing data: {self.data_path}")

        df = pd.read_csv(self.data_path)
        df_clean = df[df["fund_order_status"].isin(["SUCCESS", "FAILED"])].copy()
        df_clean["target"] = (df_clean["fund_order_status"] == "SUCCESS").astype(int)
        return df_clean

    def train(self) -> None:
        logger.info("🚀 Training Pipeline Start [Strategy: {}]", self.config.KIND)
        df = self.load_and_clean_data()
        X = pd.get_dummies(df[["fund_type", "fund_amount"]], drop_first=True)
        y = df["target"]
        self.feature_names = X.columns.tolist()

        self.model = self._model_factory()
        self.model.fit(X, y)

        self.model_path.parent.mkdir(exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.feature_names, self.features_path)
        logger.success("Model Strategy successfully trained and saved.")

    def predict(self, fund_type: str, amount: float) -> Tuple[str, float]:
        if self.model is None:
            if not self.model_path.exists():
                raise RuntimeError("Model assets not found. Run training first.")
            self.model = joblib.load(self.model_path)
            self.feature_names = joblib.load(self.features_path)

        # SECURITY FIX: Replacing 'assert' with a proper type check for Bandit
        if not isinstance(self.model, ModelStrategy):
            raise TypeError("Loaded model does not conform to ModelStrategy protocol")

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
    config = RandomForestConfig(n_estimators=100, max_depth=5)
    predictor = FundingPredictor(config=config)

    if not predictor.model_path.exists():
        predictor.train()

    logger.info("--- Axiom Engine: Online ---")
    predictor.predict("TRANSFER_P2P", 1250.0)


if __name__ == "__main__":
    main()
