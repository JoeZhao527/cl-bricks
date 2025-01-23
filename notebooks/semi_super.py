print(f"Start importing modules")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
from zipfile import ZipFile
import warnings
import pickle
import torch
from torch.utils.data import Dataset
import tsfel
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
# import lightgbm as lgb
# from hiclass import LocalClassifierPerNode, LocalClassifierPerParentNode, LocalClassifierPerLevel
# from lightgbm import LGBMClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')

print(f"Start running")

"""
Data split
"""
def create_folds(train_y, n_splits=10):
    # Create a label array for stratification
    # We'll use the first non-zero label for each row as the stratification target
    stratify_labels = []
    for _, row in train_y.iterrows():
        labels = row[train_y.columns != 'filename'].values
        # Get first non-negative label, or 0 if all negative
        first_positive = next((i for i, x in enumerate(labels) if x >= 0), 0)
        stratify_labels.append(first_positive)
    
    # Create StratifiedKFold object
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Generate fold indices
    folds = []
    for train_idx, val_idx in skf.split(train_y, stratify_labels):
        folds.append({
            'train': train_idx,
            'val': val_idx
        })
    
    return folds

# %%
train_y = pd.read_csv("../downloads/train_y_v0.1.0.csv")

# %%
folds = create_folds(train_y)

# %% [markdown]
# # Prepare features

# %% [markdown]
# ### Prepare pre-extracted features

# %%
raw_train_sets = [
    pd.read_csv("../downloads/train_data_features_v3_fixed/train_features_full_v3.csv"),
    pd.read_csv("../downloads/train_data_features_v3_fixed/train_features_split1_2_v3.csv"),
    pd.read_csv("../downloads/train_data_features_v3_fixed/train_features_split2_2_v3.csv"),
    pd.read_csv("../downloads/train_data_features_v3_fixed/train_features_split1_3_v3.csv"),
    pd.read_csv("../downloads/train_data_features_v3_fixed/train_features_split2_3_v3.csv"),
    pd.read_csv("../downloads/train_data_features_v3_fixed/train_features_split3_3_v3.csv"),
    pd.read_csv("../downloads/train_data_features_v3_fixed/train_features_split1_4_v3.csv"),
    pd.read_csv("../downloads/train_data_features_v3_fixed/train_features_split2_4_v3.csv"),
    pd.read_csv("../downloads/train_data_features_v3_fixed/train_features_split3_4_v3.csv"),
    pd.read_csv("../downloads/train_data_features_v3_fixed/train_features_split4_4_v3.csv")
]

# %%
raw_test_X = pd.read_csv("../downloads/test_features_full_v3.csv")

# %%
pre_feature_list = ['0_Absolute energy',
 '0_Area under the curve',
 '0_Autocorrelation',
 '0_Average power',
 '0_Centroid',
 '0_ECDF Percentile Count_0',
 '0_ECDF Percentile Count_1',
 '0_ECDF Percentile_0',
 '0_ECDF Percentile_1',
 '0_ECDF_0',
 '0_ECDF_1',
 '0_ECDF_2',
 '0_ECDF_3',
 '0_ECDF_4',
 '0_ECDF_5',
 '0_ECDF_6',
 '0_ECDF_7',
 '0_ECDF_8',
 '0_ECDF_9',
 '0_Entropy',
 '0_Histogram mode',
 '0_Interquartile range',
 '0_Kurtosis',
 '0_Max',
 '0_Mean',
 '0_Mean absolute deviation',
 '0_Mean absolute diff',
 '0_Mean diff',
 '0_Median',
 '0_Median absolute deviation',
 '0_Median absolute diff',
 '0_Median diff',
 '0_Min',
 '0_Negative turning points',
 '0_Neighbourhood peaks',
 '0_Peak to peak distance',
 '0_Positive turning points',
 '0_Root mean square',
 '0_Signal distance',
 '0_Skewness',
 '0_Slope',
 '0_Standard deviation',
 '0_Sum absolute diff',
 '0_Variance',
 '0_Zero crossing rate',
 '0_Fundamental frequency',
 '0_Human range energy',
 '0_Max power spectrum',
 '0_Maximum frequency',
 '0_Median frequency',
 '0_Power bandwidth',
 '0_Wavelet entropy',
 'value_median',
 'value_mean',
 'value_qmean',
 'value_max',
 'value_min',
 'value_maxmin',
 'value_diffmax',
 'value_diffmin',
 'value_diffmean',
 'value_diffqmean',
 'value_diffmedian',
 'value_diffmaxmin',
 'time_diffmean',
 'time_diffqmean',
 'time_diffmax',
 'time_diffmin',
 'time_diffmedian',
 'value_std',
 'value_var',
 'value_diffstd',
 'value_diffvar',
 'time_diffstd',
 'time_diffvar',
 'time_burstiness',
 'time_total',
 'time_event_density',
 'time_entropy',
 'time_slope'
]

# %%
train_sets = [trn[pre_feature_list] for trn in raw_train_sets]
test_X = raw_test_X[pre_feature_list]

# %% [markdown]
# # Prepare labels

# %%
def get_active_labels_np(row):
    """More efficient version using numpy"""
    arr = row.to_numpy() # convert to numpy array
    indices = np.where(arr == 1)[0] # get indices where value is 1
    labels = row.index[indices].tolist() # get labels from indices
    return labels

labelhir = train_y.apply(get_active_labels_np, axis=1).tolist()

# %%
level_labels = [list(train_y.columns[1:]), [], [], [], []]

for k in range(0, 4):
    check_labels = level_labels[k]
    label_len = len(check_labels)
    idx_is_subset_of_col = pd.DataFrame(0, index=check_labels, columns=check_labels)
    is_subset = []

    for i in tqdm(range(label_len)):
        for j in range(label_len):
            src_lb, tgt_lb = check_labels[i], check_labels[j]
            src = train_y[train_y[src_lb] == 1]
            tgt = train_y[(train_y[src_lb] == 1) & (train_y[tgt_lb] == 1)]

            idx_is_subset_of_col.loc[src_lb, tgt_lb] = len(src) <= len(tgt)
            if len(src) <= len(tgt) and src_lb != tgt_lb:
                is_subset.append([src_lb, tgt_lb])

    remove_label = set([s[0] for s in is_subset])
    
    for rl in remove_label:
        level_labels[k].remove(rl)
        level_labels[k+1].append(rl)

    print(f"Label split level {k} done.")

# %%
tiers = {
    1: level_labels[0],
    2: level_labels[1],
    3: level_labels[2],
    4: level_labels[3],
    5: level_labels[4]
}

def get_tier(label):
    for tier_num, tier_list in tiers.items():
        if label in tier_list:
            return tier_num
    return None  # Handle cases where the label isn't found in any tier

def sort_labels(labels):
    return sorted(labels, key=lambda label: (get_tier(label) or float('inf'), label))


# %%
sorted_labelhir = [sort_labels(labels) for labels in labelhir]

# %%
label_hier = np.array(
    sorted_labelhir,
    dtype=object,
)

# %%
padded_label = pd.Series(label_hier).apply(lambda x: x + ['None'] * (5 - len(x)) if len(x) < 5 else x)

# %%
# Count Nones at each level
for i in range(5):
    none_count = sum(padded_label.apply(lambda x: x[i] == 'None'))
    print(f"Level {i+1}: {none_count} None values out of {len(padded_label)} total ({none_count/len(padded_label):.2%})")

# %% [markdown]
# # Model Training

# %%
from typing import List

def train_random_forest(
    train_X: List[pd.DataFrame],
    unlabeled_X: pd.DataFrame,
    _label: np.ndarray,
    folds,
    model_class,
    params: dict,
    none_ratio_thr: float,
):
    """
    Train random forest models using k-fold cross validation
    
    Args:
        train_X: Training features DataFrame
        _label: Array of labels
        folds: List of dictionaries containing train/val indices
        
    Returns:
        tuple: (list of trained classifiers, list of scores, list of validation predictions)
    """
    classifiers = []
    val_feat_df_list = []
    unlabeled_feat_df_list = []

    _label = _label.astype(object)
    
    for f_idx, fold in enumerate(folds):
        # Prepare train and validation data for this fold
        train_X_fold_list = []
        train_y_fold_list = []
        for trn_x in train_X:
            train_X_fold_list.append(trn_x.iloc[fold['train']])
            train_y_fold_list.append(_label[fold['train']])

        train_X_fold = pd.concat(train_X_fold_list)
        train_y_fold = np.concatenate(train_y_fold_list)

        valid_X_fold_list = []
        valid_y_fold_list = []
        for trn_x in train_X:
            valid_X_fold_list.append(trn_x.iloc[fold['val']])
            valid_y_fold_list.append(_label[fold['val']])

        val_X_fold = pd.concat(valid_X_fold_list)
        val_y_fold = np.concatenate(valid_y_fold_list)
        
        # Check the train_y_fold. If more than 30% of samples are labeled "None",
        # randomly sample from the "None" to make that ratio no more than 30%.
        none_mask = (train_y_fold == "None")
        none_count = np.sum(none_mask)
        total_samples = len(train_y_fold)
        none_ratio = none_count / total_samples if total_samples > 0 else 0

        if none_ratio > none_ratio_thr:
            # Calculate how many "None" labels we should keep (30% of total)
            max_none_to_keep = int(none_ratio_thr * (total_samples - none_count))

            # Randomly choose which "None" labels to keep
            none_indices = np.where(none_mask)[0]

            # Fix the random seed before shuffling for reproducibility
            rng = np.random.RandomState(f_idx)
            rng.shuffle(none_indices)
            
            keep_none_indices = none_indices[:max_none_to_keep]

            # Indices of all non-"None" labels
            other_indices = np.where(~none_mask)[0]

            # Combine indices to keep and then sort
            new_indices = np.concatenate([keep_none_indices, other_indices])
            new_indices = np.sort(new_indices)  # Sort so we can index the DataFrame consistently

            # Subset the training data
            train_X_fold = train_X_fold.iloc[new_indices]
            train_y_fold = train_y_fold[new_indices]

            # print(f"Sampled: none-ratio: {none_ratio}, removed: {none_count - max_none_to_keep}")

        # Prepare semi-supervise dataset
        semi_train_X_fold = pd.concat([train_X_fold, unlabeled_X])
        semi_train_y_fold = np.concatenate([train_y_fold, np.array([-1] * len(unlabeled_X))])

        # Create and train Random Forest model
        estimator = model_class(**params)
        model = SelfTrainingClassifier(estimator, verbose=1)
        model.fit(semi_train_X_fold, semi_train_y_fold)
        
        classifiers.append(model)
        
        # Prepare the validation prediction results as next level input features
        val_preds = model.predict_proba(val_X_fold)
        val_pred_df = pd.DataFrame(data=val_preds, columns=model.classes_)

        val_fold_info = []
        for _f in range(len(train_X)):
            f_info = pd.DataFrame(data=fold['val'], columns=["fold_idx"])
            f_info['dataset_idx'] = _f
            val_fold_info.append(f_info)

        val_fold_idx = pd.concat(val_fold_info)
        
        val_feat_df = pd.concat([
            val_fold_idx.reset_index(drop=True),
            val_X_fold.reset_index(drop=True),
            val_pred_df,
        ], axis=1)

        val_feat_df_list.append(val_feat_df)

        # Prepare the unlabeled data prediction results as next level input features
        unlabeled_preds = model.predict_proba(unlabeled_X)
        unlabeled_pred_df = pd.DataFrame(data=unlabeled_preds, columns=model.classes_)
        
        unlabeled_feat_df = pd.concat([
            unlabeled_X.reset_index(drop=True),
            unlabeled_pred_df,
        ], axis=1)

        unlabeled_feat_df_list.append(unlabeled_feat_df)

    return classifiers, val_feat_df_list, unlabeled_feat_df_list

def setup_prev_level_prediction(predictions, fold_num, num_datasets):
    new_train_level_x = pd.concat([predictions[i] for i in range(fold_num)]).sort_values(['dataset_idx', 'fold_idx'])
    return [
        new_train_level_x[new_train_level_x['dataset_idx'] == i] \
            .drop(columns=['dataset_idx', 'fold_idx']) \
            .reset_index(drop=True)
        for i in range(num_datasets)
    ]

# %% [markdown]
# ### Train the high precision model by allowing None prediction

# %%
cliped_test_X = np.clip(test_X, a_min=None, a_max=np.finfo(np.float32).max)

# %%
prec_classifiers = []
prec_val_predictions = []
unlabeled_predictions = []

params = {
    'n_estimators': 100,
    'random_state': 42,
    'n_jobs': 20  # Use all available cores
}

model_cls = RandomForestClassifier

none_ratio_thr_list = [0.1, 0.15, 0.35, 0.75, 0.85]

# train_input = [train_X_full, train_X_1, train_X_2]
train_input = train_sets
test_input = cliped_test_X

for i in range(5):
    print(f"Training level {i}")
    _classifiers, _val_predictions, _unlabeled_preds = train_random_forest(
        train_input,
        test_input,
        np.array([x[i] for x in padded_label]),
        folds,
        params=params,
        model_class=model_cls,
        none_ratio_thr=none_ratio_thr_list[i]
    )
    prec_classifiers.append(_classifiers)
    prec_val_predictions.append(_val_predictions)
    unlabeled_predictions.append(_unlabeled_preds)

    train_input = setup_prev_level_prediction(_val_predictions, fold_num=len(folds), num_datasets=len(train_input))
    trn_col_names = list(train_input[0].columns)
    test_input = pd.concat(_unlabeled_preds, axis=1).groupby(level=0, axis=1).mean()[trn_col_names]

# %%
cliped_test_X = np.clip(test_X, a_min=None, a_max=np.finfo(np.float32).max)

# %%
def make_predictions_with_models(classifiers, test_data):
    test_preds_all = []
    for clf in tqdm(classifiers):
        pred = clf.predict_proba(test_data)
        test_preds_all.append(pd.DataFrame(data=pred, columns=clf.classes_))
    return test_preds_all

# %%
test_preds_list = []
test_input = cliped_test_X
for i in range(5):
    print(f"Predicting level {i}")
    test_preds_all = make_predictions_with_models(prec_classifiers[i], test_input)

    _level_res = pd.concat(test_preds_all, axis=1).groupby(level=0, axis=1).mean()
    test_input = pd.concat([test_input, _level_res], axis=1)
    
    test_preds_list.append(test_preds_all)

# %%
test_level_agg = []
for _level in tqdm(range(5)):
    _level_res = pd.concat(test_preds_list[_level], axis=1).groupby(level=0, axis=1).mean()
    assert not _level_res.isna().values.any()

    for col in _level_res.columns:
        _level_res = _level_res.rename(columns={col: f"{col}_{_level}"})

    test_level_agg.append(_level_res)

# %%
stacked = np.stack(
    test_level_agg[i].idxmax(axis=1).apply(lambda x: x[:-2])
    for i in range(5)
).transpose()

for row in tqdm(stacked):
    # Find first occurrence of 'None' if any
    none_idx = np.where(row == 'None')[0]
    if len(none_idx) > 0:
        # Set all elements after first None to None
        first_none = none_idx[0]
        row[first_none:] = 'None'
        
stacked

# %%
columnlist = ['Active_Power_Sensor', 'Air_Flow_Sensor',
       'Air_Flow_Setpoint', 'Air_Temperature_Sensor',
       'Air_Temperature_Setpoint', 'Alarm', 'Angle_Sensor',
       'Average_Zone_Air_Temperature_Sensor',
       'Chilled_Water_Differential_Temperature_Sensor',
       'Chilled_Water_Return_Temperature_Sensor',
       'Chilled_Water_Supply_Flow_Sensor',
       'Chilled_Water_Supply_Temperature_Sensor', 'Command',
       'Cooling_Demand_Sensor', 'Cooling_Demand_Setpoint',
       'Cooling_Supply_Air_Temperature_Deadband_Setpoint',
       'Cooling_Temperature_Setpoint', 'Current_Sensor',
       'Damper_Position_Sensor', 'Damper_Position_Setpoint', 'Demand_Sensor',
       'Dew_Point_Setpoint', 'Differential_Pressure_Sensor',
       'Differential_Pressure_Setpoint',
       'Differential_Supply_Return_Water_Temperature_Sensor',
       'Discharge_Air_Dewpoint_Sensor', 'Discharge_Air_Temperature_Sensor',
       'Discharge_Air_Temperature_Setpoint',
       'Discharge_Water_Temperature_Sensor', 'Duration_Sensor',
       'Electrical_Power_Sensor', 'Energy_Usage_Sensor',
       'Filter_Differential_Pressure_Sensor', 'Flow_Sensor', 'Flow_Setpoint',
       'Frequency_Sensor', 'Heating_Demand_Sensor', 'Heating_Demand_Setpoint',
       'Heating_Supply_Air_Temperature_Deadband_Setpoint',
       'Heating_Temperature_Setpoint', 'Hot_Water_Flow_Sensor',
       'Hot_Water_Return_Temperature_Sensor',
       'Hot_Water_Supply_Temperature_Sensor', 'Humidity_Setpoint',
       'Load_Current_Sensor', 'Low_Outside_Air_Temperature_Enable_Setpoint',
       'Max_Air_Temperature_Setpoint', 'Min_Air_Temperature_Setpoint',
       'Outside_Air_CO2_Sensor', 'Outside_Air_Enthalpy_Sensor',
       'Outside_Air_Humidity_Sensor',
       'Outside_Air_Lockout_Temperature_Setpoint',
       'Outside_Air_Temperature_Sensor', 'Outside_Air_Temperature_Setpoint',
       'Parameter', 'Peak_Power_Demand_Sensor', 'Position_Sensor',
       'Power_Sensor', 'Pressure_Sensor', 'Rain_Sensor',
       'Reactive_Power_Sensor', 'Reset_Setpoint',
       'Return_Air_Temperature_Sensor', 'Return_Water_Temperature_Sensor',
       'Room_Air_Temperature_Setpoint', 'Sensor', 'Setpoint',
       'Solar_Radiance_Sensor', 'Speed_Setpoint', 'Static_Pressure_Sensor',
       'Static_Pressure_Setpoint', 'Status', 'Supply_Air_Humidity_Sensor',
       'Supply_Air_Static_Pressure_Sensor',
       'Supply_Air_Static_Pressure_Setpoint', 'Supply_Air_Temperature_Sensor',
       'Supply_Air_Temperature_Setpoint', 'Temperature_Sensor',
       'Temperature_Setpoint', 'Thermal_Power_Sensor', 'Time_Setpoint',
       'Usage_Sensor', 'Valve_Position_Sensor', 'Voltage_Sensor',
       'Warmest_Zone_Air_Temperature_Sensor', 'Water_Flow_Sensor',
       'Water_Temperature_Sensor', 'Water_Temperature_Setpoint',
       'Wind_Direction_Sensor', 'Wind_Speed_Sensor',
       'Zone_Air_Dewpoint_Sensor', 'Zone_Air_Humidity_Sensor',
       'Zone_Air_Humidity_Setpoint', 'Zone_Air_Temperature_Sensor'
]

# %%
zipftest = ZipFile('../downloads/test_X_v0.1.0.zip', 'r')
listtestfile = zipftest.namelist()[1:]

# %%
stackedfinalresult = pd.DataFrame(columns=['filename'])
stackedfinalresult['filename'] = pd.Series(listtestfile).apply(lambda x: x.split("/")[-1])

for labelname in columnlist:
    stackedfinalresult[labelname] = 0

test_preds = stacked
for i in tqdm(range(len(test_preds))):
    # stackedfinalresult.loc[i, test_preds[i]] = 1
    predlist = test_preds[i].tolist()
    predlist = [x for x in predlist if x != 'None']
    for predlabelname in predlist:
    	stackedfinalresult.loc[i, predlabelname] = 1

# %%
stackedfinalresult = stackedfinalresult.assign(**{col: stackedfinalresult[col].astype(float) for col in stackedfinalresult.columns if col != "filename"})

# %%
def check_pred_num(_final_res, thr=0.4):
    # Exclude 'filename' column if it exists
    filtered_df = _final_res.drop(columns=['filename'], errors='ignore')

    return (filtered_df >= thr).sum(axis=1)

# %%
print("Hit num distribution")
print(check_pred_num(stackedfinalresult, thr=0.35).value_counts())

# %%
stackedfinalresult.to_csv("../logs/submit/0123_semi_supervise_dev.csv", index=False)

# %%



