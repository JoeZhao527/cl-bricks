# Run function for one model, multiple train set, one test set
from ensemble.pipeline.base_random_forest import run as base_model_run

# Run function for multiple models, multiple train set, multiple test set
from ensemble.pipeline.base_multi_model import run as ensemble_run

# Run function for multiple models stacked on previous output, multiple train set, multiple test set
from ensemble.pipeline.stacking_multi_model import run as stack_ensemble_run

# Configurations. Each one is a dict of configurations
from ensemble.config.pipeline import (
    base_rf, base_lgb, base_xgboost, weighted_rf,
    base_ensemble, stacking_ensemble
)

from copy import deepcopy

if __name__ == '__main__':
    """
    Before running, make sure update the paths in ensemble.config.paths.PATHS.
    All used data path will be defined there. 

    Current version does not save model for memory. Running complete pipeline will 
    probably stack on the local server.
    """
    # Prepare configurations
    cfg = deepcopy(base_ensemble)

    ### COMMENT OUT THE FOLLOWING FOR COMPLETE RUN
    cfg["data"]["n_train_sets"] = 2
    cfg["data"]["n_splits"] = 2

    for k, v in cfg["model"].items():
        cfg["model"][k]["model_params"]["n_estimators"] = 2

    # Start runninng
    ensemble_run(cfg)