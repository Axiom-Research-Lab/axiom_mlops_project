from pathlib import Path

import pytest

from src.main import FundingPredictor, ModelStrategy, RandomForestConfig

# --- Fixtures ---


@pytest.fixture
def default_config() -> RandomForestConfig:
    """Provides a standard Pydantic configuration for testing."""
    return RandomForestConfig(n_estimators=10, max_depth=2)


@pytest.fixture
def predictor(default_config: RandomForestConfig) -> FundingPredictor:
    """
    Fixture to provide a FundingPredictor instance.
    Now injects the required configuration (Dependency Injection).
    """
    return FundingPredictor(config=default_config)


# --- Tests ---


def test_predictor_initialization(
    predictor: FundingPredictor, default_config: RandomForestConfig
) -> None:
    """
    Given: A FundingPredictor class with a Pydantic config.
    When: It is initialized.
    Then: It should store the config and have correct paths.
    """
    assert predictor.config == default_config
    assert predictor.config.KIND == "RandomForest"
    assert "models" in str(predictor.model_path)


def test_pydantic_validation_error() -> None:
    """
    Given: An invalid configuration (negative estimators).
    When: Creating the config.
    Then: Pydantic should raise a ValidationError.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        # n_estimators has a gt=0 (greater than 0) constraint
        RandomForestConfig(n_estimators=-5)


def test_data_cleaning_logic(predictor: FundingPredictor) -> None:
    """Verify the cleaning logic still maps SUCCESS/FAILED to 1/0."""
    df = predictor.load_and_clean_data()
    assert "target" in df.columns
    assert set(df["target"].unique()).issubset({0, 1})


def test_predict_raises_error_without_model(predictor: FundingPredictor, tmp_path: Path) -> None:
    """Verify guard clause triggers when model file is missing."""
    predictor.model_path = tmp_path / "non_existent.joblib"
    predictor.model = None

    with pytest.raises(RuntimeError) as excinfo:
        predictor.predict("TRANSFER_P2P", 100.0)
    assert "Model assets not found" in str(excinfo.value)


@pytest.mark.parametrize("fund_type, amount", [("TRANSFER_P2P", 1000.0), ("CASH_DEPOSIT", 50.0)])
def test_prediction_output(predictor: FundingPredictor, fund_type: str, amount: float) -> None:
    """
    Ensure the Strategy Pattern returns valid predictions.

    Given: A predictor.
    When: We train the model and then predict.
    Then: It should return a valid result and follow the Protocol.
    """
    # 1. Train the model first so the assets exist on disk
    predictor.train()

    # 2. Now perform the prediction
    result, proba = predictor.predict(fund_type, amount)

    # 3. Assertions
    assert isinstance(predictor.model, ModelStrategy)
    assert result in ["SUCCESS", "FAILED"]
    assert 0.0 <= proba <= 1.0
