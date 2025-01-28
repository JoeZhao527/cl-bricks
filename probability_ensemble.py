import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple, List
from ensemble.config.paths import PATHS
from ensemble.config.labels import LABEL_TIERS, LEVEL_LABLES, LABEL_NAMES
from ensemble.data.label_processing import LabelTiers
from ensemble.model.predictor import get_test_agg, get_stacked_res, post_processing


def level_split(preds: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Args:
        preds (pd.DataFrame): prediction for one dataset, contains all tiers

    Returns:
        pred_list (List[pd.DataFrame]): a list of prediction of the dataset for each tier
    """
    col_tiers = [[] for _ in range(LABEL_TIERS)]
    for col in list(preds.columns):
        col_tiers[int(col[-1])].append(col)

    # Each entry in the pred_list 
    preds_list = []
    for cols in col_tiers:
        split_pred = preds[cols]
        split_pred = split_pred.rename(columns={col: col[:-2] for col in split_pred.columns})
        preds_list.append(split_pred)

    return preds_list


def check_pred_num(_final_res, thr=0.5):
    # Exclude 'filename' column if it exists
    filtered_df = _final_res.drop(columns=['filename'], errors='ignore')

    return (filtered_df >= thr).sum(axis=1)


def aggregate(pred_list: List[pd.DataFrame], train_y: pd.DataFrame, weights=None):
    if weights != None:
        raise NotImplementedError
    
    dataset_preds = [[] for _ in range(LABEL_TIERS)]
    for preds in tqdm(pred_list, desc=f"Aggregating"):
        preds_tier_split = level_split(preds)
        for i, preds_tier in enumerate(preds_tier_split):
            dataset_preds[i].append(preds_tier)
    
    print(len(dataset_preds))
    print(len(dataset_preds[0]))

    level_pred_list = get_test_agg(dataset_preds)
    stacked_res = get_stacked_res(level_pred_list)
    print(stacked_res)
    exit(0)
    final_res = post_processing(stacked_res, LABEL_NAMES, list(train_y['filename']))

    return final_res


if __name__ == '__main__':
    # UPDATE THESE PATHS FOR ENSEMBLE
    prob_prediction_paths = {
        "xgb": [
            "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/xgb/test_predictions/tst_preds_0.csv",
            # "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/xgb/test_predictions/tst_preds_1.csv",
            # "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/xgb/test_predictions/tst_preds_2.csv",
            # "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/xgb/test_predictions/tst_preds_3.csv",
            # "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/xgb/test_predictions/tst_preds_4.csv",
            # "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/xgb/test_predictions/tst_preds_5.csv",
        ],
        "lgb": [
            "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/lgb/test_predictions/tst_preds_0.csv",
            # "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/lgb/test_predictions/tst_preds_1.csv",
            # "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/lgb/test_predictions/tst_preds_2.csv",
            # "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/lgb/test_predictions/tst_preds_3.csv",
            # "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/lgb/test_predictions/tst_preds_4.csv",
            # "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/lgb/test_predictions/tst_preds_5.csv",
        ],
        "rf": [
            "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/rf/test_predictions/tst_preds_0.csv",
            # "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/rf/test_predictions/tst_preds_1.csv",
            # "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/rf/test_predictions/tst_preds_2.csv",
            # "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/rf/test_predictions/tst_preds_3.csv",
            # "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/rf/test_predictions/tst_preds_4.csv",
            # "./logs/ensemble/base_ensemble/01_27_2025-16_33_48/rf/test_predictions/tst_preds_5.csv",
        ],
    }

    train_y = pd.read_csv(PATHS.train_y_path)

    prediction_list = []
    for k, paths in prob_prediction_paths.items():
        for p in tqdm(paths, desc=f"Loading {k} results:"):
            prediction_list.append(pd.read_csv(p))
    
    final_res = aggregate(prediction_list, train_y=train_y)

    print(check_pred_num(final_res))

    arr = final_res.drop(columns=["filename"]).values
    np.save("0128_prob_ensemble.npy", np.stack(np.where(arr == 1)))
