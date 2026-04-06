from pathlib import Path

import pandas as pd
import pytest

from src.main import FundingPredictor

# --- Fixtures ---


@pytest.fixture
def predictor() -> FundingPredictor:
    """Fixture to provide a standard FundingPredictor instance."""
    return FundingPredictor()


# --- Tests ---


def test_predictor_initialization(predictor: FundingPredictor) -> None:
    """
    Given: A FundingPredictor class.
    When: It is initialized.
    Then: It should have the correct default paths and a None model.
    """
    assert "data" in str(predictor.data_path)
    assert "models" in str(predictor.model_path)
    assert predictor.model is None


def test_data_cleaning_logic(predictor: FundingPredictor) -> None:
    """
    Given: A raw dataset.
    When: load_and_clean_data is called.
    Then: The resulting DataFrame should only contain 0 and 1 in the target column.
    """
    df = predictor.load_and_clean_data()

    assert "target" in df.columns
    unique_targets = df["target"].unique()
    # Check that the logic correctly mapped SUCCESS/FAILED to 1/0
    assert set(unique_targets).issubset({0, 1})


def test_predict_raises_error_without_model(predictor: FundingPredictor, tmp_path: Path) -> None:
    """
    Given: A predictor where the model file is missing.
    When: predict() is called.
    Then: It should raise a RuntimeError due to the guard clause in src/main.py.
    """
    # Force a non-existent model path
    predictor.model_path = tmp_path / "missing_model.joblib"
    predictor.model = None

    with pytest.raises(RuntimeError) as excinfo:
        predictor.predict("TRANSFER_P2P", 100.0)

    assert "Model assets are not initialized" in str(excinfo.value)


@pytest.mark.parametrize(
    "fund_type, amount",
    [("TRANSFER_P2P", 1000.0), ("INSTANT_EFT_TOPUP", 50.0), ("CASH_DEPOSIT", 5000.0)],
)
def test_multiple_prediction_scenarios(
    predictor: FundingPredictor, fund_type: str, amount: float
) -> None:
    """
    Given: Various valid funding types and amounts.
    When: The predict method is executed.
    Then: It should return a valid result string and a probability between 0 and 1.
    """
    result, proba = predictor.predict(fund_type, amount)

    assert result in ["SUCCESS", "FAILED"]
    assert 0.0 <= proba <= 1.0
    assert isinstance(proba, float)
