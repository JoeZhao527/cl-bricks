from .class_weight import base_rf_performance_weights, base_xgb_performance_weights

base_rf = {
    "data": {
        "n_splits": 10, # controls number of cross validation
        "random_state": 42,
    },
    "model": {
        "model_cls": "random_forest",
        "model_params": {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': 20
        },
        "none_ratio_thr_list": [0.1, 0.15, 0.35, 0.75, 0.85],
        "model_save_dir": "base_rf_models"  # this will be the directory inside `output_base`
    },
    "output_base": "./logs/ensemble/base_rf"
}

base_lgb = {
    "data": {
        "n_splits": 10,
        "random_state": 42,
    },
    "model": {
        "model_cls": "lightgbm",
        "model_params": {
            'verbose':-1,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': 20,
            'objective': 'multiclass',
        },
        "none_ratio_thr_list": [0.1, 0.15, 0.35, 0.75, 0.85],
        "model_save_dir": "base_lgb_models"  # this will be the directory inside `output_base`
    },
    "output_base": "./logs/ensemble/base_lgb"
}

base_xgboost = {
    "data": {
        "n_splits": 10,
        "random_state": 42,
    },
    "model": {
        "model_cls": "xgboost",
        "model_params": {
            'device': 'cuda',
            'n_estimators': 400,       # Number of trees
            'learning_rate': 0.3,     # Default learning rate
            'max_depth': 6,           # Maximum depth of trees
            'min_child_weight': 1,    # Minimum sum of weights in a child node
            'subsample': 0.8,         # Fraction of samples per tree
            'colsample_bytree': 0.8,  # Fraction of features per tree
            'gamma': 0,               # Minimum loss reduction for split
            'reg_alpha': 0,           # L1 regularization term
            'reg_lambda': 1,          # L2 regularization term
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_jobs': 20
        },
        "none_ratio_thr_list": [0.1, 0.15, 0.35, 0.75, 0.85],
        "model_save_dir": "base_xgb_models"  # this will be the directory inside `output_base`
    },
    "output_base": "./logs/ensemble/base_xgboost"
}

base_ensemble = {
    "data": {
        "n_splits": 10,
        "random_state": 42
    },
    "model": {
        "xgb": base_xgboost["model"],
        "lgb": base_lgb["model"],
        "rf": base_rf["model"],
    },
    "output_base": "./logs/ensemble/base_ensemble"
}

stacking_ensemble = {
    "data": {
        "n_splits": 10,
        "random_state": 42,
        "prev_stage": {
            "xgb": "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/xgb",
            "lgb": "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/lgb",
            "rf": "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/rf"
        }
    },
    "model": {
        "xgb": base_xgboost["model"],
        "lgb": base_lgb["model"],
        "rf": base_rf["model"],
    },
    "output_base": "./logs/ensemble/stack_ensemble"
}

weighted_rf = {
    "data": {
        "n_splits": 10,
        "random_state": 42,
    },
    "model": {
        "model_cls": "random_forest",
        "model_params": {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': 20,
            'class_weight': base_rf_performance_weights
        },
        "none_ratio_thr_list": [0.1, 0.15, 0.35, 0.75, 0.85],
    },
    "output_base": "./logs/ensemble/weighted_rf"
}

weighted_xgboost = {
    "data": {
        "n_splits": 10,
        "random_state": 42,
    },
    "model": {
        "model_cls": "xgboost",
        "model_params": {
            'device': 'cuda',
            'n_estimators': 400,       # Number of trees
            'learning_rate': 0.3,     # Default learning rate
            'max_depth': 6,           # Maximum depth of trees
            'min_child_weight': 1,    # Minimum sum of weights in a child node
            'subsample': 0.8,         # Fraction of samples per tree
            'colsample_bytree': 0.8,  # Fraction of features per tree
            'gamma': 0,               # Minimum loss reduction for split
            'reg_alpha': 0,           # L1 regularization term
            'reg_lambda': 1,          # L2 regularization term
            'random_state': 42,
            'eval_metric': 'logloss',
            'class_weight': base_xgb_performance_weights,
            'n_jobs': 20
        },
        "none_ratio_thr_list": [0.1, 0.15, 0.35, 0.75, 0.85],
    },
    "output_base": "./logs/ensemble/base_xgboost"
}

weighted_ensemble = {
    "data": {
        "n_splits": 10,
        "random_state": 42
    },
    "model": {
        "rf": weighted_rf["model"],
        "xgb": weighted_xgboost["model"],
    },
    "output_base": "./logs/ensemble/weighted_ensemble"
}