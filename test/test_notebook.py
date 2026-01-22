import pytest
import pandas as pd
import numpy as np
import sys
import importlib.util
from pathlib import Path
from sklearn.metrics import f1_score

# --------------------------------------------------------------------------------------------
# Fixtures (Required to load the agent's code and data)
# --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def predictor_class():
    """Dynamically imports PrenatalRiskClassifier from /results/utils.py"""
    source_path = Path("/results/utils.py")
    if not source_path.exists():
        pytest.fail(f"CRITICAL: Source file '{source_path}' missing.")
    try:
        spec = importlib.util.spec_from_file_location("utils_module", source_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["utils_module"] = module
        spec.loader.exec_module(module)
        return module.PrenatalRiskClassifier
    except AttributeError:
        pytest.fail("CRITICAL: Class 'PrenatalRiskClassifier' not found in utils.py")
    except Exception as e:
        pytest.fail(f"CRITICAL: Import failed. {e}")

@pytest.fixture(scope="module")
def student_data():
    """Loads blind test features and hidden ground truth."""
    X_test_path = Path("/workspace/data/test.csv")
    y_test_path = Path("/eval_data/ground_truth.csv")
    train_path = Path("/workspace/data/train.csv")

    if not all(p.exists() for p in [X_test_path, y_test_path, train_path]):
        pytest.fail("CRITICAL: Data missing. Check /eval_data for ground truth.")

    # Load Data
    X_test = pd.read_csv(X_test_path)

    # Target is now 'fetal_health' based on the Prenatal Risk Classification task
    y_test = pd.read_csv(y_test_path)["fetal_health"]

    df_train = pd.read_csv(train_path)

    X_train = df_train.drop("fetal_health", axis=1)
    y_train = df_train["fetal_health"]

    if "fetal_health" in X_test.columns:
        pytest.fail("SECURITY: Target leaked in test.csv")

    return X_train, y_train, X_test, y_test

# --------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------

def test_performance_macro_f1(predictor_class, student_data):
    """Test 1: Macro F1-score must be >= 0.92 on blind test set."""
    PrenatalRiskClassifier = predictor_class
    X_train, y_train, X_test, y_test = student_data

    model = PrenatalRiskClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Ensure predictions are 1D
    y_pred = np.asarray(y_pred).ravel()

    macro_f1 = f1_score(y_test, y_pred, average="macro")

    assert macro_f1 >= 0.92, f"FAIL: Macro F1-score is {macro_f1:.4f}, expected >= 0.92"

def test_robustness_garbage_columns(predictor_class, student_data):
    """Test 2: Model must handle extra 'garbage' columns without crashing."""
    PrenatalRiskClassifier = predictor_class
    X_train, y_train, X_test, _ = student_data

    model = PrenatalRiskClassifier()
    model.fit(X_train, y_train)

    # Create a copy of test data with a random garbage column
    X_dirty = X_test.copy()
    X_dirty["random_garbage_999"] = np.random.random(len(X_dirty))

    try:
        model.predict(X_dirty)
    except Exception as e:
        pytest.fail(f"FAIL: Model crashed when input contained extra columns. Error: {e}")

def test_visualization_plot_counts_verification(predictor_class, student_data):
    """
    Test 3: Verify that predicted fetal_health counts are computed correctly
    (i.e., the values used in the bar plot must match value_counts from predictions).
    """
    PrenatalRiskClassifier = predictor_class
    X_train, y_train, X_test, _ = student_data

    model = PrenatalRiskClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = pd.Series(np.asarray(y_pred).ravel(), name="fetal_health")

    # Compute counts that SHOULD be used for the plot
    counts = y_pred.value_counts().sort_index()

    # Basic sanity checks
    assert len(y_pred) == len(X_test), "FAIL: Number of predictions does not match test rows."
    assert counts.sum() == len(X_test), "FAIL: Count total does not match number of predictions."
    assert counts.shape[0] >= 2, "FAIL: Only one class predicted; plot would be meaningless."

    # Optional verification if the student provides a helper method for plot counts
    # (Not required, but supported if implemented)
    if hasattr(model, "get_plot_counts"):
        try:
            student_counts = model.get_plot_counts(X_test)
            student_counts = pd.Series(student_counts).sort_index()
            pd.testing.assert_series_equal(counts, student_counts, check_names=False)
        except Exception as e:
            pytest.fail(f"FAIL: Plot counts verification failed. Error: {e}")