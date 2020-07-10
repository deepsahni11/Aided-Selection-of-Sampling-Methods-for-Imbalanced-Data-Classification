# Aided-Selection-of-Sampling-Methods-for-Imbalanced-Data-Classification



## List of files
* data_creation.py -> create synthetic datasets
* data_resampling.py -> apply resampling methods on the imbalanced datasets
* meta_features_generate.py -> generate meta-features for each dataset
* classifier_predictions.py -> calculates probability predictions for resampled data
* prediction_recall_classifier_predictions.py -> calculates precision-recall measures from probability predictions
* generate_bootstrap_samples.py -> creates bootstrap samples: X (meta-features) & Y (precision-recall measures)
* precision_regressor_predictions_rmse.py -> calculates rmse scores for precision predictions of regressor models for all sampling methods
* recall_regressor_predictions_rmse.py -> calculates rmse scores for precision predictions of regressor models for all sampling methods
* precision_regressor_predictions.py -> regressor model predicts precision scores for every sampling method, probability threshold and dataset combination
* recall_regressor_predictions.py -> regressor model predicts recall scores for every sampling method, probability threshold and dataset combination
