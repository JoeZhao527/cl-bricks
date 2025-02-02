# BBB Multilabel Classification

## Getting Start
### Data Preparation
The following data needs to be prepared / downloaded / preprocessed before starts

- `train_y_v0.1.0.csv`: labels for each training sample, downloaded from https://www.aicrowd.com/challenges/brick-by-brick-2024/dataset_files
- `test_X_v0.1.0.zip`: raw signals for each testing sample, downloaded from https://www.aicrowd.com/challenges/brick-by-brick-2024/dataset_files
- `test_data_features_v3_fixed/`: a directory that contains all the preprocessed test features
- `train_data_features_v3_fixed/`: a directory that contains all the preprocessed train features

### Training and Inferencing Demo
After preprocessing the data, try a small demo of training and inferencing pipeline by running:
```
python main.py config/demo.py
```

This will produce an output directory like this:
```
./logs/ensemble/base_ensemble/02_02_2025-23_15_42/    # Timestamped folder for logs and results
│
├── rf/                                               # Random Forest models and prediction results
│   ├── models/                                      # Model checkpoints for each tier and fold
│   │   ├── tier_0_fold_0.pkl
│   │   ├── tier_0_fold_1.pkl
│   │   ├── tier_1_fold_0.pth
│   │   └── ...
│   │
│   ├── test_predictions/                            # Test set predictions from Random Forest
│   │   ├── tst_preds_0.csv
│   │   └── ...                                      # Will have multiple prediction when using test data augmentation
│
├── xgb/                                              # XGBoost models and prediction results
│   ├── models/                                      # Model checkpoints for each tier and fold
│   │   ├── tier_0_fold_0.pkl
│   │   ├── tier_0_fold_1.pkl
│   │   ├── tier_1_fold_0.pth
│   │   └── ...
│   │
│   ├── test_predictions/                            # Test set predictions from XGBoost
│   │   ├── tst_preds_0.csv
│   │   └── ...                                      # Will have multiple prediction when using test data augmentation
│
└── final_prediction.csv                              # Ensembled results for submission
```

To inference with the trained models, update the `test_zip_path` and `test_x_paths` in the configuration yaml, then run the script with 2 additional arguments:
```
# NOTICE: REPLACE THE model_dir WITH YOUR OWN OUTPUT PATH
python main.py config/demo.py config/demo.py --skip_train --model_dir=./logs/ensemble/base_ensemble/02_02_2025-23_15_42
```

## Further Configurations
### Project Structure
```
main.py                     # Entry point for the application
ensemble/                   # Core module for model pipeline
│
├── data/                   # Data processing functions (feature normalization, feature crossing, tier labeling)
│
├── model/                  # Functions for model training and prediction
│
├── pipeline.py             # Data loading, processing, model training, validation, and prediction pipeline
│
└── probability_ensemble.py # Ensemble model to aggregate predictions from different models
```

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
