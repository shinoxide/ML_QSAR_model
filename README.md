# ML_QSAR_Model

This repository, `ML_QSAR_model`, contains Python scripts for developing machine learning-based QSAR (Quantitative Structure-Activity Relationship) classification models.

## Scripts Overview

1. **`1a_generate_ecfp.py`**  
   This script generates Extended-Connectivity Fingerprints (ECFP4) with a diameter of 4 bonds using RDKit, an open-source cheminformatics toolkit.

2. **`1b_generate_ecfp.py`**  
   A memory-efficient version of `1a_generate_ecfp.py` that can handle multiple SDF files.

3. **`2_splitting_with_structural_similarity.py`**  
   This script uses RDKit for splitting the dataset, ensuring stratified sampling based on both structural features and activity labels.

4. **`3_rf_hyperparameter_tuning.py`**  
   Utilizes `GridSearchCV` from Scikit-learn to systematically evaluate a predefined grid of hyperparameter values for Random Forest classifiers. The optimal hyperparameter combination is identified through cross-validation.

5. **`4_svm_hyperparameter_tuning.py`**  
   Similar to the previous script, but for Support Vector Machine (SVM) classifiers. It also uses `GridSearchCV` to find the best hyperparameters.

6. **`5_model.py`**  
   This script runs the model using optimized and default parameters, plots the average ROC AUC over cross-validation iterations, computes various evaluation metrics, and organizes the results into directories.

## Getting Started

To run these scripts, ensure that:

- The required dependencies, including RDKit and Scikit-learn, are installed.
- The correct data is being read from the appropriate directories.

Properly organizing your data and verifying the input paths will ensure the scripts run smoothly.
