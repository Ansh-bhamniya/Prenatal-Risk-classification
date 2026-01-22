# Prenatal Risk Classification

You are given a dataset containing cardiotocography (CTG) features used to assess fetal well-being in the directory `data/`. The directory consists of:
* **train.csv**: Contains columns `['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 'light_decelerations', 'severe_decelerations', 'prolongued_decelerations', 'abnormal_short_term_variability', 'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance', 'histogram_tendency', 'fetal_health']`.
* **test.csv**: Contains only the features `['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 'light_decelerations', 'severe_decelerations', 'prolongued_decelerations', 'abnormal_short_term_variability', 'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance', 'histogram_tendency']`. **The target 'fetal_health' column has been removed for evaluation.**

**Goal**: Achieve a **macro F1-score ≥ 0.91** on the hidden test dataset.  
**Goal (Visualization)**: Generate a clear bar chart showing **count vs predicted `fetal_health`** classes using predictions from `test.csv`.

### Deliverables

Complete the class `PrenatalRiskClassifier` in the initial notebook. Once the model is complete, **write the entire class** along with necessary imports into the file `/results/utils.py`. Make sure to create the directory `/results` if it does not exist.

* Your class must handle `fit(X, y)` and `predict(X)` taking DataFrames as input.
* **Robustness**: Your model must automatically ignore any extra or "garbage" columns present in the input DataFrame during prediction.
* **Feature Engineering Requirement**: You must create **at least 2 new engineered features** from the existing features. These features should be meaningful and improve model performance. Examples include: ratios, interactions, polynomial features, or domain-specific transformations. The feature engineering must be implemented within your class and applied consistently during both training and prediction.
* **Hyperparameter Optimization Requirement**: Your model must perform hyperparameter optimization (e.g., using GridSearchCV, RandomizedSearchCV, or Optuna) during the `fit()` method. The optimization should use cross-validation (at least 3-fold) to select optimal hyperparameters. You must store the best parameters found.
* **Cross-Validation Requirement**: Implement at least 3-fold cross-validation during training to ensure robust model performance. The cross-validation should be used for hyperparameter selection.
* **Feature Importance Analysis**: Your class must provide a method `get_feature_importance()` that returns a pandas Series or dictionary mapping feature names to their importance scores. The importance scores should be normalized (sum to 1.0) and sorted in descending order.
* **Prediction Probabilities**: Your class must provide a method `predict_proba(X)` that returns prediction probabilities for each class, with shape `(n_samples, n_classes)`.
* **Per-Class Metrics**: Your class must provide a method `evaluate_per_class_metrics(X, y)` that returns a dictionary containing precision, recall, and F1-score for each class.
* **Visualization Requirements**: 
  - After generating predictions for `test.csv`, you must create a bar graph plot of **count vs predicted `fetal_health`** (i.e., the number of samples in each predicted class).
  - Create a confusion matrix visualization.
  - Create a feature importance bar plot showing the top 10 most important features.
* **Verification Requirement**: You must verify the counts used in the bar graph plot directly from your predicted `fetal_health` values (e.g., using `value_counts()` or an equivalent method) and ensure the plotted counts exactly match the computed counts.
* **Robustness Requirements**: Your model must handle:
  - Extra "garbage" columns
  - Missing values in input data
  - Data type inconsistencies (e.g., numeric columns as strings)

The model will be instantiated and tested automatically:

    ```python
    from utils import PrenatalRiskClassifier

    model = PrenatalRiskClassifier()  # Ensure default arguments work
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    feature_importance = model.get_feature_importance()
    ```