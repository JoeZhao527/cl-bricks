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


def aggregate(pred_list: List[pd.DataFrame], filenames: list, weights=None):
    if weights != None:
        raise NotImplementedError
    
    dataset_preds = [[] for _ in range(LABEL_TIERS)]
    for preds in tqdm(pred_list, desc=f"Aggregating"):
        preds_tier_split = level_split(preds)
        for i, preds_tier in enumerate(preds_tier_split):
            dataset_preds[i].append(preds_tier)

    level_pred_list = get_test_agg(dataset_preds)

    stacked_res = get_stacked_res(level_pred_list)

    final_res = post_processing(stacked_res, LABEL_NAMES, filenames)

    return final_res


def get_norm_weightes(reports: List[Tuple]):
    # List of reports and their corresponding model names
    reports = [
        (pd.read_csv(report), f"{model_name}")
        for model_name, report in reports
    ]

    model_cols = [r[1] for r in reports]

    # Initialize the merged DataFrame with the first report
    norm_weight = reports[0][0][['col', 'f1']].rename(columns={'f1': reports[0][1]})

    # Merge the remaining reports in a loop
    for report, col_name in reports[1:]:
        norm_weight = pd.merge(
            norm_weight,
            report[['col', 'f1']].rename(columns={'f1': col_name}),
            on=['col']
        )

    # Calculate the sum of weights for normalization
    weight_sum = norm_weight[model_cols].sum(axis=1)

    # Normalize the weights
    for col in model_cols:
        norm_weight[col] = norm_weight[col] / weight_sum

    return {
        model_name: dict(norm_weight[['col', model_name]].values)
        for model_name, _ in reports
    }

if __name__ == '__main__':
    # UPDATE THESE PATHS FOR ENSEMBLE
    prob_prediction_paths = {
        "lgb": [
            "./logs/ensemble/base_ensemble/02_01_2025-13_25_10/lgb/test_predictions/tst_preds_0.csv",
            "./logs/ensemble/base_ensemble/02_01_2025-13_25_10/lgb/test_predictions/tst_preds_1.csv",
            "./logs/ensemble/base_ensemble/02_01_2025-13_25_10/lgb/test_predictions/tst_preds_2.csv",
            "./logs/ensemble/base_ensemble/02_01_2025-13_25_10/lgb/test_predictions/tst_preds_3.csv",
            "./logs/ensemble/base_ensemble/02_01_2025-13_25_10/lgb/test_predictions/tst_preds_4.csv",
            "./logs/ensemble/base_ensemble/02_01_2025-13_25_10/lgb/test_predictions/tst_preds_5.csv",
        ],
        "rf": [
            "./logs/ensemble/base_ensemble/02_01_2025-13_25_10/rf/test_predictions/tst_preds_0.csv",
            "./logs/ensemble/base_ensemble/02_01_2025-13_25_10/rf/test_predictions/tst_preds_1.csv",
            "./logs/ensemble/base_ensemble/02_01_2025-13_25_10/rf/test_predictions/tst_preds_2.csv",
            "./logs/ensemble/base_ensemble/02_01_2025-13_25_10/rf/test_predictions/tst_preds_3.csv",
            "./logs/ensemble/base_ensemble/02_01_2025-13_25_10/rf/test_predictions/tst_preds_4.csv",
            "./logs/ensemble/base_ensemble/02_01_2025-13_25_10/rf/test_predictions/tst_preds_5.csv",
        ],
        "xgb": [
            "./logs/ensemble/base_ensemble/02_01_2025-13_25_10/xgb/test_predictions/tst_preds_0.csv",
            "./logs/ensemble/base_ensemble/02_01_2025-13_25_10/xgb/test_predictions/tst_preds_1.csv",
            "./logs/ensemble/base_ensemble/02_01_2025-13_25_10/xgb/test_predictions/tst_preds_2.csv",
            "./logs/ensemble/base_ensemble/02_01_2025-13_25_10/xgb/test_predictions/tst_preds_3.csv",
            "./logs/ensemble/base_ensemble/02_01_2025-13_25_10/xgb/test_predictions/tst_preds_4.csv",
            "./logs/ensemble/base_ensemble/02_01_2025-13_25_10/xgb/test_predictions/tst_preds_5.csv"
        ]
    }

    report_paths = {
        "rf": "logs/ensemble/base_ensemble/02_01_2025-13_25_10/rf/cv_report.csv",
        "xgb": "logs/ensemble/base_ensemble/02_01_2025-13_25_10/xgb/cv_report.csv",
        "lgb": "logs/ensemble/base_ensemble/02_01_2025-13_25_10/lgb/cv_report.csv",
    }

    norm_weights = get_norm_weightes(report_paths)
    weighted: bool = True

    from zipfile import ZipFile
    zipftest = ZipFile(PATHS.test_zip_path, 'r')
    listtestfile = list(zipftest.namelist()[1:])


    prediction_list = []
    for k, paths in prob_prediction_paths.items():
        for p in tqdm(paths, desc=f"Loading {k} results:"):
            model_res = pd.read_csv(p)
            if weighted:
                for col in model_res:
                    model_res[col] = model_res[col] * norm_weights[k].get(col[:-2], 1/len(report_paths))
            prediction_list.append(model_res)
    
    final_res = aggregate(prediction_list, listtestfile)

    print(check_pred_num(final_res).value_counts())

    # UPDATE THE OUTPUT CSV PATH. THIS CAN BE DIRECTLY SUBMITTED.
    # final_res.to_csv("0129_prob_ensemble_xgb_rf_only.csv", index=False)

    # compression the csv to index of the binary array
    arr = final_res.drop(columns=["filename"]).values

    # UPDATE THE PATH FOR COMPRESSED RES
    np.save("0201_lgb_rf_xgb_weighted.npy", np.stack(np.where(arr == 1)))
