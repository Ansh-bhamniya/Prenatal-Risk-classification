# Prenatal Risk Classification

You are given a dataset containing cardiotocography (CTG) features used to assess fetal well-being in the directory `data/`. The directory consists of:
* **train.csv**: Contains columns `['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 'light_decelerations', 'severe_decelerations', 'prolongued_decelerations', 'abnormal_short_term_variability', 'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance', 'histogram_tendency', 'fetal_health']`.
* **test.csv**: Contains only the features `['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 'light_decelerations', 'severe_decelerations', 'prolongued_decelerations', 'abnormal_short_term_variability', 'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance', 'histogram_tendency']`. **The target 'fetal_health' column has been removed for evaluation.**

**Goal**: Achieve a **macro F1-score ≥ 0.92** on the hidden test dataset.  
**Goal (Visualization)**: Generate a clear bar chart showing **count vs predicted `fetal_health`** classes using predictions from `test.csv`.

### Deliverables

Complete the class `PrenatalRiskClassifier` in the initial notebook. Once the model is complete, **write the entire class** along with necessary imports into the file `/results/utils.py`. Make sure to create the directory `/results` if it does not exist.

* Your class must handle `fit(X, y)` and `predict(X)` taking DataFrames as input.
* **Robustness**: Your model must automatically ignore any extra or "garbage" columns present in the input DataFrame during prediction.
* **Visualization Requirement**: After generating predictions for `test.csv`, you must create a bar graph plot of **count vs predicted `fetal_health`** (i.e., the number of samples in each predicted class).
* **Verification Requirement**: You must verify the counts used in the bar graph plot directly from your predicted `fetal_health` values (e.g., using `value_counts()` or an equivalent method) and ensure the plotted counts exactly match the computed counts.

The model will be instantiated and tested automatically:

    ```python
    from utils import PrenatalRiskClassifier

    model = PrenatalRiskClassifier()  # Ensure default arguments work
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ```