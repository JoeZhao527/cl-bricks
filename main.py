from ensemble.pipeline.base_random_forest import run as base_model_run
from ensemble.pipeline.base_multi_model import run as ensemble_run
from ensemble.pipeline.stacking_multi_model import run as stack_ensemble_run
from ensemble.config.pipeline import base_rf, base_lgb, base_xgboost, base_ensemble, stacking_ensemble, weighted_rf, weighted_ensemble
from copy import deepcopy

if __name__ == '__main__':
    cfg = deepcopy(weighted_ensemble)
    # cfg["data"]["n_train_sets"] = 2
    # cfg["data"]["n_splits"] = 2

    # # cfg["model"]["model_params"]["n_estimators"] = 2

    # for k, v in cfg["model"].items():
    #     cfg["model"][k]["model_params"]["n_estimators"] = 2

    # stack_ensemble_run(cfg)
    ensemble_run(cfg)