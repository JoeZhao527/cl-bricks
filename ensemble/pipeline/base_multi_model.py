import pandas as pd
import numpy as np
from functools import partial
import datetime
from zipfile import ZipFile
import os

from ensemble.config.feature_names import FEATURE_NAMES
from ensemble.config.labels import LABEL_TIERS, LABEL_NAMES
from ensemble.config.paths import PATHS
from ensemble.model.model_zoo import model_zoo
from ensemble.data.feature_processing import bin_normalization, feature_crossing, train_test_wrapper, train_test_list_wrapper
from ensemble.data.label_processing import LabelTiers
from ensemble.model.trainer import BaseModel
from ensemble.model.predictor import test_result_pipeline

def log(msg):
    print(f"[{datetime.datetime.now()}] Main Log: {msg}")

def run(cfg):
    # INITIALIZE OUTPUT DIRECTORIES
    output_base = os.path.join(cfg.get("output_base"), f"{datetime.datetime.now().strftime('%m_%d_%Y-%H_%M_%S')}")
    os.makedirs(output_base)
    log(f"Output dir: {output_base}")

    # DATA PROCESSING
    # Prepare train label
    label_tier = LabelTiers(PATHS.train_y_path)
    log(f"Label tier initialization done")

    # split folds
    folds = label_tier.create_folds(cfg["data"]["n_splits"], random_state=cfg["data"]["random_state"])
    log(f"Split to {len(folds)} folds")

    # Load and prepare input features
    log(f"Start to load training data")
    train_sets = [pd.read_csv(p) for p in PATHS.train_x_paths]

    log(f"Start to load testing data")
    test_sets = [pd.read_csv(p) for p in PATHS.test_x_paths]
    
    if cfg["data"].get("n_train_sets", None):
        train_sets = train_sets[:cfg["data"]["n_train_sets"]]
    
    log(f"Using {len(train_sets)} train sets. Original column size {len(train_sets[0].columns)}.")

    # only keep the selected feature columns
    # train_sets = [trn[FEATURE_NAMES] for trn in train_sets]
    # test_sets = [tst[FEATURE_NAMES] for tst in test_sets]
    # test_X = test_X[FEATURE_NAMES]
    
    log(f"Got {len(train_sets[0].columns)} feature columns after selection.")

    # normalization
    train_sets, test_sets = train_test_list_wrapper(
        train_sets, test_sets,
        partial(bin_normalization, bin_nums=1000)
    )
    log(f"Got {len(train_sets[0].columns)} feature columns after normalization.")

    # feature crossing
    train_sets, test_sets = train_test_list_wrapper(
        train_sets, test_sets,
        partial(feature_crossing, corr_thr=0.2)
    )
    log(f"Got {len(train_sets[0].columns)} feature columns after feature crossing.")

    # get test filenames
    zipftest = ZipFile(PATHS.test_zip_path, 'r')
    test_filenames = zipftest.namelist()[1:]
    log(f"Loaded test filenames")

    # MODEL TRAINING
    # Initialize trainer
    for base_model_name, base_model_cfg in cfg["model"].items():
        _output_dir = os.path.join(output_base, base_model_name)
        os.makedirs(_output_dir)
        
        model_save_dir = None
        if base_model_cfg.get("model_save_dir", None):
            model_save_dir = os.path.join(_output_dir, base_model_cfg["model_save_dir"])

        model = BaseModel(
            model_cls=model_zoo.get(base_model_cfg["model_cls"]),
            model_params=base_model_cfg["model_params"],
            none_ratio_thr_list=base_model_cfg["none_ratio_thr_list"],
            model_save_dir=model_save_dir,
            folds=folds,
            train_input=train_sets,
            padded_labels=label_tier.padded_labels,
        )
        log(f"Model intialization completed.")

        # model training
        model.train()
        log(f"Model training completed.")

        # evaluation
        cv_report = model.evaluation(label_tier.train_y)
        cv_report_path = os.path.join(_output_dir, "cv_report.csv")
        cv_report.to_csv(cv_report_path, index=False)

        # Save validation result
        val_pred_res = model.get_validation_preds()
        val_pred_res_path = os.path.join(_output_dir, "val_preds.csv")
        val_pred_res.to_csv(val_pred_res_path, index=False)
        log(f"Validation prediction saved to {val_pred_res_path}")

        # PREDICTION
        test_output_dir = os.path.join(_output_dir, "test_predictions")
        os.makedirs(test_output_dir)

        for tst_idx, test_X in enumerate(test_sets):
            test_prediction, final_result = test_result_pipeline(
                classifiers=model.classifiers,
                cliped_test_X=test_X,
                columnlist=LABEL_NAMES,
                listtestfile=test_filenames
            )
            log(f"Test inferencing completed, saving the results")

            test_prediction_path = os.path.join(test_output_dir, f"tst_preds_{tst_idx}.csv")
            test_prediction.to_csv(test_prediction_path, index=False)
            log(f"Test prediction result saved to {test_prediction_path}")

            final_result_path = os.path.join(test_output_dir, f"final_result_{tst_idx}.csv")
            final_result.to_csv(final_result_path, index=False)
            log(f"Final result saved to {final_result_path}")
        
    log(f"Finished.")