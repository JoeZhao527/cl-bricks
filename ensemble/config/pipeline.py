base_rf = {
    "data": {
        "n_splits": 10,
    },
    "model": {
        "model_cls": "random_forest",
        "model_params": {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': 8
        },
        "none_ratio_thr_list": [0.1, 0.15, 0.35, 0.75, 0.85],
    },
    "output_base": "./logs/ensemble/base_rf"
}

base_lgb = {
    "data": {
        "n_splits": 10,
    },
    "model": {
        "model_cls": "lightgbm",
        "model_params": {
            'verbose':-1,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': 8,  # Use all available cores
            'objective': 'multiclass',
        },
        "none_ratio_thr_list": [0.1, 0.15, 0.35, 0.75, 0.85],
    },
    "output_base": "./logs/ensemble/base_lgb"
}

base_xgboost = {
    "data": {
        "n_splits": 10,
    },
    "model": {
        "model_cls": "xgboost",
        "model_params": {
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
            'n_jobs': 8
        },
        "none_ratio_thr_list": [0.1, 0.15, 0.35, 0.75, 0.85],
    },
    "output_base": "./logs/ensemble/base_xgboost"
}

base_ensemble = {
    "data": {
        "n_splits": 10,
    },
    "model": {
        "rf": base_rf["model"],
        "lgb": base_lgb["model"],
        "xgb": base_xgboost["model"]
    },
    "output_base": "./logs/ensemble/base_ensemble"
}