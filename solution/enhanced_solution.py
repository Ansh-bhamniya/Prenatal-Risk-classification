{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Fetal Health Prediction\n",
    "\n",
    "This notebook implements the data processing, modeling, and evaluation pipeline for classifying fetal health status using Cardiotocography (CTG) data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import json\n",
    "import os\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import f1_score, roc_auc_score"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Loading"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load datasets\n",
    "try:\n",
    "    medical_df = pd.read_csv(\"data/medical_data.csv\")\n",
    "    histogram_df = pd.read_csv(\"data/histogram_data.csv\")\n",
    "except FileNotFoundError:\n",
    "    # Fallback for local testing\n",
    "    medical_df = pd.read_csv(\"medical_data.csv\")\n",
    "    histogram_df = pd.read_csv(\"histogram_data.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Data Preparation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature Engineering\n",
    "EPS = 1e-6\n",
    "\n",
    "# 1. MajorDecelBurden\n",
    "medical_df['MajorDecelBurden'] = medical_df['severe_decelerations'] + medical_df['prolongued_decelerations']\n",
    "\n",
    "# 2. VariabilityAbnormalityIndex\n",
    "medical_df['VariabilityAbnormalityIndex'] = (\n",
    "    medical_df['abnormal_short_term_variability'] + \n",
    "    medical_df['percentage_of_time_with_abnormal_long_term_variability']\n",
    ")\n",
    "\n",
    "# 3. Reassurance Features\n",
    "medical_df['TotalDecelerations'] = (\n",
    "    medical_df['light_decelerations'] + \n",
    "    medical_df['severe_decelerations'] + \n",
    "    medical_df['prolongued_decelerations']\n",
    ")\n",
    "medical_df['ReassuranceRatio'] = medical_df['accelerations'] / (medical_df['TotalDecelerations'] + EPS)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Dataset Integration & Cleaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Combine on patient_id\n",
    "final_df = pd.merge(medical_df, histogram_df, on='patient_id')\n",
    "\n",
    "# Filter health_insurance (remove 0 or False)\n",
    "# Ensuring boolean logic covers both numeric 0 and boolean False\n",
    "final_df = final_df[(final_df['health_insurance'] != 0) & (final_df['health_insurance'] != False)]\n",
    "\n",
    "# Drop patient_id\n",
    "final_df = final_df.drop(columns=['patient_id'])\n",
    "\n",
    "# Drop rows with NA\n",
    "final_df = final_df.dropna()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Model Setup & Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define X and y\n",
    "X = final_df.drop(columns=['fetal_health'])\n",
    "y = final_df['fetal_health']\n",
    "\n",
    "# Split dataset (70/30, seed 42)\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.30, random_state=42\n",
    ")\n",
    "\n",
    "# Initialize and Fit Random Forest\n",
    "clf = RandomForestClassifier(random_state=42)\n",
    "clf.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Predictions\n",
    "y_pred = clf.predict(X_test)\n",
    "y_prob = clf.predict_proba(X_test)\n",
    "\n",
    "# Metrics\n",
    "f1 = f1_score(y_test, y_pred, average='macro')\n",
    "auc = roc_auc_score(y_test, y_prob, multi_class='ovr')\n",
    "\n",
    "print(f\"F1 Score (Macro): {f1}\")\n",
    "print(f\"AUC Score: {auc}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Deliverables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Feature Importance Dictionary\n",
    "importances = clf.feature_importances_\n",
    "feature_names = X.columns\n",
    "feature_importance_dict = {\n",
    "    feat: round(imp, 5) \n",
    "    for feat, imp in zip(feature_names, importances)\n",
    "}\n",
    "\n",
    "# 2. Model Quality\n",
    "model_quality = {\n",
    "    \"f1\": round(f1, 5),\n",
    "    \"auc\": round(auc, 5)\n",
    "}\n",
    "\n",
    "# 3. Fetal Status (Serialized DataFrame of Counts)\n",
    "unique, counts = np.unique(y_pred, return_counts=True)\n",
    "fetal_status_df = pd.DataFrame({'class': unique, 'count': counts})\n",
    "fetal_status = fetal_status_df.to_dict(orient='split')\n",
    "\n",
    "print(\"Model Quality:\", model_quality)\n",
    "print(\"Fetal Status:\", fetal_status)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# === CRITICAL: SAVE VARIABLES FOR TEST HARNESS ===\n",
    "\n",
    "# Define the variables to save\n",
    "notebook_vars = {\n",
    "    \"feature_importance_dict\": feature_importance_dict,\n",
    "    \"model_quality\": model_quality,\n",
    "    \"fetal_status\": fetal_status\n",
    "}\n",
    "\n",
    "# Ensure directory exists (handles cloud/docker environments)\n",
    "verifier_dir = \"/logs/verifier\"\n",
    "if not os.path.exists(verifier_dir):\n",
    "    try:\n",
    "        os.makedirs(verifier_dir)\n",
    "    except PermissionError:\n",
    "        # Fallback for local testing if /logs is root-protected\n",
    "        verifier_dir = \".\"\n",
    "\n",
    "# Save the JSON\n",
    "with open(f\"{verifier_dir}/notebook_variables.json\", \"w\") as f:\n",
    "    json.dump(notebook_vars, f)\n",
    "\n",
    "print(f\"Variables saved to {verifier_dir}/notebook_variables.json\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}