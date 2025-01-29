# BBB Multilabel Classification

## Getting Start
### Data Preparation
The following data needs to be prepared / downloaded / preprocessed before starts

- `train_y_v0.1.0.csv`: labels for each training sample, downloaded from https://www.aicrowd.com/challenges/brick-by-brick-2024/dataset_files
- `test_X_v0.1.0.zip`: raw signals for each testing sample, downloaded from https://www.aicrowd.com/challenges/brick-by-brick-2024/dataset_files
- `test_data_features_v3_fixed/`: a directory that contains all the preprocessed test features
- `train_data_features_v3_fixed/`: a directory that contains all the preprocessed train features

### Model Training
Update `main.py` to select different configuration. Start training with `python main.py`. Intermediate results and final results will be saved to `logs/`.

## Further Configurations
### Project Structure
`main.py`: entry point.

`probability_ensemble.py`: collect and aggregating the prediction results. After running `python main.py`, run this to prepare ensemble and prepare the final submission file. Input and output paths need to be changed to your own paths in this file when using it.

`ensemble/`: contains the code for model training, validation and prediction.

`ensemble/config`: pipeline configuration files

`ensemble/data`: feature (normalization and feature crossing) and label (tier definition) processing functions. 

`ensemble/model`: model training and prediction functions.

`ensemble/pipeline`: data loading, processing, model training, validation and prediction pipeline. Each `.py` under this directory is a pipeline, the 0.575 f1 score pipeline is in `base_multi_model.py`.

### Configurations
`ensemble/config/feature_names.py`
```
FEATURE_NAMES: base feature columns that will be used for train and test. Additional columns were dynamically appended during training.
```

`ensemble/config/labels.py`
```
LABEL_TIERS: Number of tiers, fixed to 5.

LABEL_NAMES: All label names that needs to be predicted in the BBB dataset

LEVEL_LABLES: A list of list contains labels in each tier
```

`ensemble/config/paths.py`
```
PATHS: A class contains all the input data file paths.
```

`ensemble/config/pipeline.py`
```
base_rf: baseline single random forest, use it with ensemble/pipeline/base_random_forest.py

base_lgb: baseline single lightgbm, use it with ensemble/pipeline/base_random_forest.py

base_xgb: baseline single xgboost, use it with ensemble/pipeline/base_random_forest.py

base_ensemble: ensemble of the three models, use it with ensemble/pipeline/base_multi_model.py
```
