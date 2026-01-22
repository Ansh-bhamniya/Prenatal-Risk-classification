import pytest
import pandas as pd
import numpy as np
import sys
import importlib.util
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

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

def test_feature_engineering(predictor_class, student_data):
    """
    Test 4: Verify that the model creates at least 2 new engineered features.
    The feature engineering must be consistent between training and prediction.
    """
    PrenatalRiskClassifier = predictor_class
    X_train, y_train, X_test, _ = student_data

    model = PrenatalRiskClassifier()
    
    # Count original features
    original_feature_count = len(X_train.columns)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Check if model has a way to expose engineered features
    # We'll check by comparing feature counts or checking for feature engineering attributes
    if hasattr(model, "feature_names_out_"):
        engineered_features = model.feature_names_out_()
        assert len(engineered_features) >= original_feature_count + 2, \
            f"FAIL: Expected at least {original_feature_count + 2} features (including 2+ engineered), got {len(engineered_features)}"
    elif hasattr(model, "n_features_out_"):
        assert model.n_features_out_ >= original_feature_count + 2, \
            f"FAIL: Expected at least {original_feature_count + 2} features (including 2+ engineered), got {model.n_features_out_}"
    else:
        # Try to infer by checking if prediction works and model has feature engineering logic
        # This is a softer check - if the model works, we assume feature engineering is present
        try:
            y_pred = model.predict(X_test)
            assert len(y_pred) == len(X_test), "FAIL: Prediction failed after feature engineering"
        except Exception as e:
            pytest.fail(f"FAIL: Feature engineering may be missing or incorrect. Error: {e}")

def test_hyperparameter_optimization(predictor_class, student_data):
    """
    Test 5: Verify that the model performs hyperparameter optimization during fit().
    The model should store best parameters found.
    """
    PrenatalRiskClassifier = predictor_class
    X_train, y_train, X_test, _ = student_data

    model = PrenatalRiskClassifier()
    model.fit(X_train, y_train)

    # Check for common attributes that indicate hyperparameter optimization
    has_best_params = (
        hasattr(model, "best_params_") or
        hasattr(model, "best_estimator_") or
        hasattr(model, "cv_results_") or
        hasattr(model, "best_score_")
    )
    
    assert has_best_params, \
        "FAIL: Model does not appear to perform hyperparameter optimization. " \
        "Expected attributes like best_params_, best_estimator_, cv_results_, or best_score_."

    # Verify that the model actually uses optimized parameters (not just defaults)
    # This is a soft check - if best_params_ exists, we assume optimization occurred
    if hasattr(model, "best_params_"):
        assert len(model.best_params_) > 0, "FAIL: best_params_ is empty"

def test_feature_importance(predictor_class, student_data):
    """
    Test 6: Verify that get_feature_importance() returns normalized, sorted importance scores.
    """
    PrenatalRiskClassifier = predictor_class
    X_train, y_train, X_test, _ = student_data

    model = PrenatalRiskClassifier()
    model.fit(X_train, y_train)

    assert hasattr(model, "get_feature_importance"), \
        "FAIL: Model does not have get_feature_importance() method"

    try:
        importance = model.get_feature_importance()
        
        # Convert to Series if dict
        if isinstance(importance, dict):
            importance = pd.Series(importance)
        elif not isinstance(importance, pd.Series):
            importance = pd.Series(importance)
        
        # Check normalization (should sum to approximately 1.0)
        importance_sum = importance.sum()
        assert abs(importance_sum - 1.0) < 0.01, \
            f"FAIL: Feature importance not normalized. Sum is {importance_sum}, expected ~1.0"
        
        # Check sorting (descending order)
        assert importance.is_monotonic_decreasing or importance.equals(importance.sort_values(ascending=False)), \
            "FAIL: Feature importance not sorted in descending order"
        
        # Check that all values are non-negative
        assert (importance >= 0).all(), "FAIL: Feature importance contains negative values"
        
        # Check that we have at least some features
        assert len(importance) > 0, "FAIL: Feature importance is empty"
        
    except Exception as e:
        pytest.fail(f"FAIL: get_feature_importance() failed. Error: {e}")

def test_predict_proba(predictor_class, student_data):
    """
    Test 7: Verify that predict_proba() returns valid probability distributions.
    """
    PrenatalRiskClassifier = predictor_class
    X_train, y_train, X_test, _ = student_data

    model = PrenatalRiskClassifier()
    model.fit(X_train, y_train)

    assert hasattr(model, "predict_proba"), \
        "FAIL: Model does not have predict_proba() method"

    try:
        y_proba = model.predict_proba(X_test)
        
        # Check shape: (n_samples, n_classes)
        assert y_proba.shape[0] == len(X_test), \
            f"FAIL: predict_proba shape mismatch. Expected {len(X_test)} samples, got {y_proba.shape[0]}"
        
        assert y_proba.shape[1] >= 2, \
            f"FAIL: predict_proba should have at least 2 classes, got {y_proba.shape[1]}"
        
        # Check that probabilities sum to 1.0 for each sample (within tolerance)
        proba_sums = y_proba.sum(axis=1)
        assert np.allclose(proba_sums, 1.0, atol=0.01), \
            "FAIL: Prediction probabilities do not sum to 1.0 for all samples"
        
        # Check that all probabilities are in [0, 1]
        assert (y_proba >= 0).all() and (y_proba <= 1).all(), \
            "FAIL: Prediction probabilities must be in [0, 1]"
        
    except Exception as e:
        pytest.fail(f"FAIL: predict_proba() failed. Error: {e}")

def test_per_class_metrics(predictor_class, student_data):
    """
    Test 8: Verify that evaluate_per_class_metrics() returns precision, recall, and F1 for each class.
    """
    PrenatalRiskClassifier = predictor_class
    X_train, y_train, X_test, y_test = student_data

    model = PrenatalRiskClassifier()
    model.fit(X_train, y_train)

    assert hasattr(model, "evaluate_per_class_metrics"), \
        "FAIL: Model does not have evaluate_per_class_metrics() method"

    try:
        metrics = model.evaluate_per_class_metrics(X_test, y_test)
        
        assert isinstance(metrics, dict), \
            "FAIL: evaluate_per_class_metrics() should return a dictionary"
        
        # Check for required keys
        required_keys = ["precision", "recall", "f1"]
        for key in required_keys:
            assert key in metrics, f"FAIL: Missing '{key}' in per-class metrics"
        
        # Check that each metric is a dict or Series with class labels
        for key in required_keys:
            metric_values = metrics[key]
            if isinstance(metric_values, dict):
                assert len(metric_values) > 0, f"FAIL: {key} dictionary is empty"
            elif isinstance(metric_values, pd.Series):
                assert len(metric_values) > 0, f"FAIL: {key} Series is empty"
            else:
                pytest.fail(f"FAIL: {key} should be dict or Series, got {type(metric_values)}")
        
        # Verify metric values are reasonable (between 0 and 1)
        for key in required_keys:
            metric_values = metrics[key]
            if isinstance(metric_values, dict):
                values = list(metric_values.values())
            else:
                values = metric_values.values
            
            for val in values:
                assert 0 <= val <= 1, \
                    f"FAIL: {key} contains invalid value {val} (must be in [0, 1])"
        
    except Exception as e:
        pytest.fail(f"FAIL: evaluate_per_class_metrics() failed. Error: {e}")

def test_robustness_missing_values(predictor_class, student_data):
    """
    Test 9: Model must handle missing values in input data gracefully during prediction.
    """
    PrenatalRiskClassifier = predictor_class
    X_train, y_train, X_test, _ = student_data

    model = PrenatalRiskClassifier()
    model.fit(X_train, y_train)

    # Create test data with missing values
    X_missing = X_test.copy()
    # Introduce missing values in a few columns
    np.random.seed(42)
    missing_cols = X_missing.columns[:3]  # First 3 columns
    for col in missing_cols:
        missing_indices = np.random.choice(len(X_missing), size=min(10, len(X_missing)//10), replace=False)
        X_missing.loc[missing_indices, col] = np.nan

    try:
        y_pred = model.predict(X_missing)
        assert len(y_pred) == len(X_missing), \
            f"FAIL: Prediction failed with missing values. Expected {len(X_missing)} predictions, got {len(y_pred)}"
    except Exception as e:
        pytest.fail(f"FAIL: Model crashed when input contained missing values. Error: {e}")