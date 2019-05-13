This folder has 3 subfolders:

data : original and generated csvs 

plots: predicted vs real values for models in 2_ml_regression_1_high_low

rdata: models and tuning parameters for various tunegrids

The remainging files are described below:

BlackFriday_Report: report on strategies and a walkthrough the training process

f_partition: a function to split data into train and test sets

load_libraries: libraries that are used throughout the scripts

regression_metrics: functions for calculating regression metrics

1_feature_engineering: feature engineering script (unused versions are commented out)

2_ml_regression_0_baseline: model training based on baseline data cleaning/engineering

2_ml_regression_1_high_low: model training based on version 1 data cleaning/engineering
(low frequency factor values are grouped into "high purchase others" and "low purchase others"

2_ml_regression_2_all_interactions: linear model training including all interaction terms

2_ml_regression_3_NA_0s: model training based on version 3 data cleaning/engineering
(missing values set to zero)

3_caret_tuning_regression: parameter tuning for all models

4_black_friday_test_predict: generates final predictions based on final model