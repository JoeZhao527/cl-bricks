from tqdm import tqdm
import pandas as pd
import numpy as np
from ensemble.config.labels import LABEL_TIERS
from datetime import datetime
from typing import Tuple


def make_predictions_with_models(classifiers, test_data, level_id):
    test_preds_all = []
    for clf in tqdm(classifiers, desc=f"[{datetime.now()}] predicting level {level_id}"):
        pred = clf.predict_proba(test_data)
        test_preds_all.append(
            pd.DataFrame(data=pred, columns=[f"{col}_{level_id}" for col in clf.get_class_names()])
        )
    return test_preds_all

def get_pred_list(classifiers, test_input):
    test_preds_list = []
    for i in range(LABEL_TIERS):
        test_preds_all = make_predictions_with_models(classifiers[i], test_input, level_id=i)
        _level_res = pd.concat(test_preds_all, axis=1).groupby(level=0, axis=1).mean()

        test_input = pd.concat([test_input, _level_res], axis=1)
        
        test_preds_list.append([df.rename(columns={col: col[:-2] for col in df.columns}) for df in test_preds_all])

    return test_preds_list

def get_test_agg(test_preds_list):
    test_level_agg = []
    for _level in tqdm(range(LABEL_TIERS), desc=f"[{datetime.now()}] Aggregating predictions"):
        _level_res = pd.concat(test_preds_list[_level], axis=1).groupby(level=0, axis=1).mean()
        assert not _level_res.isna().values.any()

        for col in _level_res.columns:
            _level_res = _level_res.rename(columns={col: f"{col}_{_level}"})

        test_level_agg.append(_level_res)

    return test_level_agg

def get_stacked_res(test_level_agg):
    stacked = np.stack(
        test_level_agg[i].idxmax(axis=1).apply(lambda x: x[:-2])
        for i in range(LABEL_TIERS)
    ).transpose()

    for row in tqdm(stacked, desc=f"[{datetime.now()}] Postprocessing Nones"):
        # Find first occurrence of 'None' if any
        none_idx = np.where(row == 'None')[0]
        if len(none_idx) > 0:
            # Set all elements after first None to None
            first_none = none_idx[0]
            row[first_none:] = 'None'
            
    return stacked

def post_processing(test_preds, columnlist, listtestfile):
    stackedfinalresult = pd.DataFrame(columns=['filename'])
    stackedfinalresult['filename'] = pd.Series(listtestfile).apply(lambda x: x.split("/")[-1])

    for labelname in columnlist:
        stackedfinalresult[labelname] = 0

    for i in tqdm(range(len(test_preds)), desc=f"[{datetime.now()}] Preparing final result file"):
        predlist = test_preds[i].tolist()
        predlist = [x for x in predlist if x != 'None']
        for predlabelname in predlist:
            stackedfinalresult.loc[i, predlabelname] = 1

    stackedfinalresult = stackedfinalresult.assign(**{col: stackedfinalresult[col].astype(float) for col in stackedfinalresult.columns if col != "filename"})
    
    return stackedfinalresult

def test_result_pipeline(classifiers, cliped_test_X, columnlist, listtestfile) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_preds_list = get_pred_list(classifiers, cliped_test_X)
    test_level_agg = get_test_agg(test_preds_list)
    test_preds = get_stacked_res(test_level_agg)
    
    final_result = post_processing(test_preds, columnlist, listtestfile)

    return pd.concat(test_level_agg, axis=1), final_result
