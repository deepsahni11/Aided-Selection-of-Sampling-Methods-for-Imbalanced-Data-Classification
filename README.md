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



## Framework of the Architecture

1.  Synthetic Data Creation

```
Create synthetic datasets using the parameters from make_classification python package as given below in the table: 

```
Metrics | Values |
--- | --- | 
 Samples per data set   | 1000  | 
 Flip fraction (noise level) | 0, 0.002, 0.004, 0.006, 0.008, 0.01 |  
 Number of features (n)  | 4, 6, 8, 9, 10, 11, 13, 15 | 
 Informative features | n-2, n-1, n  |
 Class separation | 0.3, 0.65, 1.0, 1.35, 1.7 |
 Number of clusters per class | 1, 2 for $n$ = 4 and 1, 2, 3 for rest |
 Number of classes | 2|
 Imbalance ratio between the two classes | 0.9:0.1, 0.8:0.2, 0.7:0.3, 0.6:0.4 |
 
 2.  Re-sampling of the imbalanced datasets
 
 ```
 
 ```
![alt text](https://github.com/deepsahni11/Aided-Selection-of-Sampling-Methods-for-Imbalanced-Data-Classification/blob/master/Framework_draft_final.jpg?raw=true)

