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
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.svm import SVC
from collections import defaultdict
warnings.filterwarnings('ignore')

def create_folds(train_y, n_splits=10):
    # Create a label array for stratification
    # We'll use the first non-zero label for each row as the stratification target
    stratify_labels = []
    for _, row in tqdm(train_y.iterrows(), total=len(train_y)):
        labels = row[train_y.columns != 'filename'].values
        stratify_labels.append(str(labels))
    
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

def get_active_labels_np(row):
    """More efficient version using numpy"""
    arr = row.to_numpy() # convert to numpy array
    indices = np.where(arr == 1)[0] # get indices where value is 1
    labels = row.index[indices].tolist() # get labels from indices
    return labels

def build_tree(onto):
    """
    Build a tree so that each term has at most one parent.
    The parent is determined by the longest existing term that is a substring of the child.
    """
    # Sort terms by length so that broader terms are processed (and assigned) first
    sorted_onto = sorted(onto, key=len)
    
    # Dictionaries for storing parent-child relationships
    parent_map = {}             # term -> parent
    children_map = defaultdict(list)  # parent -> [children]

    processed = []
    
    for term in sorted_onto:
        # Find all processed terms that are substrings of 'term'
        potential_parents = [p for p in processed if p in term]
        
        if not potential_parents:
            # No parent found; this term is at the root
            parent_map[term] = None
            children_map[None].append(term)
        else:
            # Pick the longest parent (closest match)
            parent = max(potential_parents, key=len)
            parent_map[term] = parent
            children_map[parent].append(term)
        
        processed.append(term)
    
    return parent_map, children_map

"""
Re-built hierachical labels
"""
LEVEL_LABELS = [[], [], [], [], []]

def print_tree(children_map, root=None, depth=0):
    """
    Recursively print the tree structure with indentation.
    'root=None' means we are listing top-level (root) terms first.
    """
    
    if root is None:
        # For all top-level terms
        for child in sorted(children_map[root]):
            print_tree(children_map, child, depth)
    else:
        # print("  " * depth + root)
        LEVEL_LABELS[depth].append(root)
        for child in sorted(children_map[root]):
            print_tree(children_map, child, depth + 1)

def train_svm_classifier(train_X, _label, folds, drop_none=False):
    """
    Train SVM models using k-fold cross validation

    Args:
        train_X: Training features DataFrame
        _label: Array of labels
        folds: List of dictionaries containing train/val indices
        drop_none: Whether to drop samples with "None" labels

    Returns:
        tuple: (list of trained classifiers, list of scores, list of validation predictions)
    """
    classifiers = []
    scores = []
    val_predictions = []  # List to store validation predictions

    # Define LightGBM parameters
    params = {
        'verbose': False,
        'random_state': 42,
    }

    for f_idx, fold in enumerate(folds):
        # Prepare train and validation data for this fold
        train_X_fold = train_X[fold['train']]
        train_y_fold = _label[fold['train']]
        val_X_fold = train_X[fold['val']]
        val_y_fold = _label[fold['val']]

        if drop_none:
            # Remove samples with "None" labels from training set
            train_mask = train_y_fold != "None"
            train_X_fold = train_X_fold[train_mask]
            train_y_fold = train_y_fold[train_mask]

            # Remove samples with "None" labels from validation set
            val_mask = val_y_fold != "None"
            val_X_fold = val_X_fold[val_mask]
            val_y_fold = val_y_fold[val_mask]
            print(f"Dropped train: {len(train_X_fold) - sum(train_mask)}, val: {len(val_X_fold) - sum(val_mask)}")

        # Check the train_y_fold. If more than 30% of samples are labeled "None",
        # randomly sample from the "None" to make that ratio no more than 30%.
        none_mask = (train_y_fold == "None")
        none_count = np.sum(none_mask)
        total_samples = len(train_y_fold)
        none_ratio = none_count / total_samples if total_samples > 0 else 0

        if none_ratio > 0.4:
            # Calculate how many "None" labels we should keep (30% of total)
            max_none_to_keep = int(0.4 * (total_samples - none_count))

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
            train_X_fold = train_X_fold[new_indices]
            train_y_fold = train_y_fold[new_indices]

            print(f"Sampled: none-ratio: {none_ratio}, removed: {total_samples - max_none_to_keep}")

        # Create and train LightGBM model
        model = SVC(**params)
        model.fit(train_X_fold, train_y_fold)

        classifiers.append(model)

        # Calculate score and save predictions on validation set
        val_preds = model.predict(val_X_fold)
        score = np.mean(val_preds == val_y_fold)
        scores.append(score)
        val_predictions.append({
            'true_labels': val_y_fold,
            'predicted_labels': val_preds,
            'fold_indices': fold['val']
        })
        print(f"Fold score: {score:.4f}")

    print(f"Average score: {np.mean(scores)}")
    return classifiers, scores, val_predictions

def make_predictions_with_models(classifiers, test_data):
    """
    Make probability predictions using multiple classifier models
    
    Args:
        classifiers: List of trained classifier models
        test_data: Test data to make predictions on
        
    Returns:
        List of probability predictions from each classifier
    """
    test_preds_all = []
    for clf in tqdm(classifiers):
        pred = clf.predict(test_data)
        test_preds_all.append(pred)
    return test_preds_all

def align_and_combine_predictions(classifiers, test_preds_all, test_data, threshold=0.0):
    """
    Aligns predictions from multiple classifiers and combines them through averaging
    
    Args:
        classifiers: List of trained classifier models
        test_preds_all: List of probability predictions from each classifier
        test_data: Test data used for predictions
        threshold: Minimum probability threshold for making predictions
        
    Returns:
        Final class predictions after aligning and combining probabilities
    """
    # Get the common classes across all classifiers
    all_classes = classifiers[0].classes_
    test_preds_aligned = []

    # Make predictions with each fold's model and align them 
    for i, clf in tqdm(enumerate(classifiers)):
        pred = test_preds_all[i]
        # Create a mapping to align predictions with common classes
        pred_dict = {_cls: idx for idx, _cls in enumerate(clf.classes_)}
        aligned_pred = np.zeros((len(test_data), len(all_classes)))

        for i, _cls in enumerate(all_classes):
            if _cls in pred_dict:
                aligned_pred[:, i] = pred[:, pred_dict[_cls]]
        
        test_preds_aligned.append(aligned_pred)

    # Stack and average the aligned predictions
    test_preds_all = np.stack(test_preds_aligned)
    test_preds_proba = test_preds_all.mean(axis=0)

    # Get max probabilities for each prediction
    max_probs = np.max(test_preds_proba, axis=1)
    
    # Convert probabilities to class predictions, using threshold
    test_preds = np.array(['None'] * len(test_data), dtype=object)
    confident_mask = max_probs >= threshold
    test_preds[confident_mask] = all_classes[np.argmax(test_preds_proba[confident_mask], axis=1)]
    
    return test_preds

# Mode along axis 0 (columns)
def mode_along_axis(arr, axis):
    return np.array([np.unique(arr[:, i], return_counts=True) for i in range(arr.shape[axis])])

if __name__ == '__main__':
    train_y_path = "../downloads/train_y_v0.1.0.csv"
    feature_path = "../prediction.pt"
    test_x_path = "../downloads/test_X_v0.1.0.zip"
    dev = True
    n_fold_cv = 2

    train_y = pd.read_csv(train_y_path)

    # prepare feature
    feature = torch.load(feature_path)
    feature = torch.concat(feature, dim=0)

    trn_feat = feature[:len(train_y)].numpy()
    tst_feat = feature[len(train_y):].numpy()

    if dev:
        train_y = train_y.iloc[:500]
        trn_feat = trn_feat[:500]
        tst_feat = tst_feat[:500]

    print(f"train shape: {trn_feat.shape}")
    print(f"test shape: {tst_feat.shape}")

    labelhir = train_y.apply(get_active_labels_np, axis=1).tolist()
    
    # Get a tier dict
    ontology_list = list(train_y.columns[1:])

    parent_map, children_map = build_tree(ontology_list)
    print_tree(children_map)

    tiers = {
        i+1: LEVEL_LABELS[i] for i in range(5)
    }

    def get_tier(label):
        for tier_num, tier_list in tiers.items():
            if label in tier_list:
                return tier_num
        return None  # Handle cases where the label isn't found in any tier

    def sort_labels(labels):
        return sorted(labels, key=lambda label: (get_tier(label) or float('inf'), label))
    
    sorted_labelhir = [sort_labels(labels) for labels in labelhir]

    label_hier = np.array(
        sorted_labelhir,
        dtype=object,
    )

    padded_label = pd.Series(label_hier).apply(lambda x: x + ['None'] * (5 - len(x)) if len(x) < 5 else x)

    # Count Nones at each level
    for i in range(5):
        none_count = sum(padded_label.apply(lambda x: x[i] == 'None'))
        print(f"Level {i+1}: {none_count} None values out of {len(padded_label)} total ({none_count/len(padded_label):.2%})")

    # train
    folds = create_folds(train_y, n_splits=n_fold_cv)

    prec_svm_classifiers = []
    prec_scores = []
    prec_svm_val_predictions = []

    for i in range(5):
        print(f"Training level {i}")
        _classifiers, _scores, _val_predictions = train_svm_classifier(trn_feat, np.array([x[i] for x in padded_label]), folds, drop_none=False)
        prec_svm_classifiers.append(_classifiers)
        prec_scores.append(_scores)
        prec_svm_val_predictions.append(_val_predictions)

    test_preds = []
    for i in range(5):
        print(f"Predicting level {i}")
        test_preds_all = make_predictions_with_models(prec_svm_classifiers[i], tst_feat)
        # print(test_preds_all)
        # test_preds.append(align_and_combine_predictions(prec_svm_classifiers[i], test_preds_all, tst_feat))
        test_preds = np.apply_along_axis(
            lambda x: np.unique(x, return_counts=True)[0][np.argmax(np.unique(x, return_counts=True)[1])],
            axis=0, arr=np.stack(test_preds_all, axis=0)
        )
    print(np.stack(test_preds_all, axis=0))
    print(test_preds)
    # Convert to array and process None values
    # stacked = np.stack(test_preds).transpose()
    stacked = test_preds.transpose()
    for row in tqdm(stacked):
        # Find first occurrence of 'None' if any
        none_idx = np.where(row == 'None')[0]
        if len(none_idx) > 0:
            # Set all elements after first None to None
            first_none = none_idx[0]
            row[first_none:] = 'None'
    
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

    zipftest = ZipFile(test_x_path, 'r')

    listtestfile = zipftest.namelist()[1:]

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

    stackedfinalresult = stackedfinalresult.assign(**{col: stackedfinalresult[col].astype(float) for col in stackedfinalresult.columns if col != "filename"})
    stackedfinalresult.to_csv("./svm_cl_feat.csv", index=False)