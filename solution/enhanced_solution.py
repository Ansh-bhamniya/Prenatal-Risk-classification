"""
Enhanced PrenatalRiskClassifier Solution
This file demonstrates the complete implementation meeting all new requirements.
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score


class PrenatalRiskClassifier:
    def __init__(self):
        """Initialize fetal health classification model with feature engineering, 
        hyperparameter optimization, and comprehensive evaluation capabilities."""
        self.target_name = "fetal_health"

        # Explicitly define expected features for robustness
        self.features = [
            'baseline value',
            'accelerations',
            'fetal_movement',
            'uterine_contractions',
            'light_decelerations',
            'severe_decelerations',
            'prolongued_decelerations',
            'abnormal_short_term_variability',
            'mean_value_of_short_term_variability',
            'percentage_of_time_with_abnormal_long_term_variability',
            'mean_value_of_long_term_variability',
            'histogram_width',
            'histogram_min',
            'histogram_max',
            'histogram_number_of_peaks',
            'histogram_number_of_zeroes',
            'histogram_mode',
            'histogram_mean',
            'histogram_median',
            'histogram_variance',
            'histogram_tendency'
        ]
        
        # Store best parameters and grid search results
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.best_estimator_ = None

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from existing features.
        Returns DataFrame with original + engineered features."""
        X_eng = X.copy()
        
        # Feature 1: Deceleration ratio (total decelerations normalized)
        deceleration_cols = ['light_decelerations', 'severe_decelerations', 'prolongued_decelerations']
        if all(col in X_eng.columns for col in deceleration_cols):
            X_eng['total_decelerations'] = (
                X_eng['light_decelerations'] + 
                X_eng['severe_decelerations'] + 
                X_eng['prolongued_decelerations']
            )
            baseline_safe = X_eng['baseline value'].replace(0, np.nan)
            X_eng['deceleration_baseline_ratio'] = X_eng['total_decelerations'] / (baseline_safe + 1e-6)
        else:
            X_eng['total_decelerations'] = 0.0
            X_eng['deceleration_baseline_ratio'] = 0.0
        
        # Feature 2: Variability ratio (short-term vs long-term variability)
        if 'mean_value_of_short_term_variability' in X_eng.columns and \
           'mean_value_of_long_term_variability' in X_eng.columns:
            long_term_safe = X_eng['mean_value_of_long_term_variability'].replace(0, np.nan)
            X_eng['variability_ratio'] = (
                X_eng['mean_value_of_short_term_variability'] / (long_term_safe + 1e-6)
            )
        else:
            X_eng['variability_ratio'] = 0.0
        
        # Feature 3: Histogram spread (range normalized by mean)
        if all(col in X_eng.columns for col in ['histogram_min', 'histogram_max', 'histogram_mean']):
            histogram_range = X_eng['histogram_max'] - X_eng['histogram_min']
            mean_safe = X_eng['histogram_mean'].replace(0, np.nan)
            X_eng['histogram_spread_ratio'] = histogram_range / (mean_safe + 1e-6)
        else:
            X_eng['histogram_spread_ratio'] = 0.0
        
        return X_eng

    def _prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select expected features, ignore extra columns, create missing columns, 
        and apply feature engineering."""
        Xc = X.copy()

        # Convert string columns to numeric where possible (robustness)
        for col in Xc.columns:
            if Xc[col].dtype == 'object':
                try:
                    Xc[col] = pd.to_numeric(Xc[col], errors='coerce')
                except:
                    pass

        # Ignore extra / garbage columns safely
        keep_cols = [c for c in self.features if c in Xc.columns]
        Xc = Xc[keep_cols].copy()

        # Ensure all expected feature columns exist
        for c in self.features:
            if c not in Xc.columns:
                Xc[c] = np.nan

        # Preserve the correct column order
        Xc = Xc[self.features]
        
        # Apply feature engineering
        Xc = self._engineer_features(Xc)

        return Xc

    def fit(self, X, y):
        """Fit the model with hyperparameter optimization using cross-validation."""
        Xp = self._prepare_X(X)
        y_series = pd.Series(y).squeeze()
        
        # Create base pipeline
        base_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(random_state=42, class_weight="balanced"))
        ])
        
        # Define hyperparameter grid
        param_grid = {
            'model__n_estimators': [200, 300, 400],
            'model__max_depth': [10, 15, 20, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
        
        # Perform grid search with 3-fold cross-validation
        grid_search = GridSearchCV(
            base_pipeline,
            param_grid,
            cv=3,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(Xp, y_series)
        
        # Store results
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.cv_results_ = grid_search.cv_results_
        self.best_estimator_ = grid_search.best_estimator_
        self.pipeline = grid_search.best_estimator_
        
        # Store feature names for feature importance
        self.feature_names_out_ = list(Xp.columns)
        self.n_features_out_ = len(self.feature_names_out_)
        
        return self

    def predict(self, X):
        """Predict fetal health classes."""
        Xp = self._prepare_X(X)
        return self.pipeline.predict(Xp)

    def predict_proba(self, X):
        """Return prediction probabilities for each class."""
        Xp = self._prepare_X(X)
        return self.pipeline.predict_proba(Xp)

    def get_feature_importance(self):
        """Return normalized feature importance scores sorted in descending order."""
        if not hasattr(self, 'pipeline'):
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Get feature importance from the model
        model = self.pipeline.named_steps['model']
        importance = model.feature_importances_
        
        # Create Series with feature names
        importance_series = pd.Series(
            importance,
            index=self.feature_names_out_
        )
        
        # Normalize to sum to 1.0
        importance_series = importance_series / importance_series.sum()
        
        # Sort in descending order
        importance_series = importance_series.sort_values(ascending=False)
        
        return importance_series

    def evaluate_per_class_metrics(self, X, y):
        """Return precision, recall, and F1-score for each class."""
        y_pred = self.predict(X)
        y_true = pd.Series(y).squeeze()
        
        # Get unique classes
        classes = sorted(np.unique(np.concatenate([y_true.unique(), y_pred])))
        
        # Calculate metrics per class
        precision = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
        
        # Create dictionary with results
        metrics = {
            'precision': dict(zip(classes, precision)),
            'recall': dict(zip(classes, recall)),
            'f1': dict(zip(classes, f1))
        }
        
        return metrics

    def get_plot_counts(self, X):
        """Return the exact counts used for the bar plot based on predicted fetal_health."""
        y_pred = pd.Series(self.predict(X), name="fetal_health")
        return y_pred.value_counts().sort_index()
