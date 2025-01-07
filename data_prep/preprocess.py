import pandas as pd
import numpy as np
import pickle
import tsfel
import os
from typing import Tuple, List, Dict, Any
from tqdm import tqdm
from zipfile import ZipFile
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

def timestamp_feature_extraction(timestamp: np.ndarray) -> Dict[str, Any]:
    """
    Extract time related feature
    """
    timestamp = timestamp - timestamp[0]

    time_diff = np.diff(timestamp)
    time_diff = time_diff[~np.isnan(time_diff)]

    time_diffmean = time_diff.mean()
    time_diffqmean = np.mean([np.percentile(time_diff, [25, 75])])
    time_diffmax = time_diff.max()
    time_diffmin = time_diff.min()
    time_diffmedian = np.median(time_diff)

    time_diffstd = np.std(time_diff)
    time_diffvar = np.var(time_diff)

    mean_diff = time_diffmean
    std_diff = time_diffstd
    time_burstiness = (std_diff - mean_diff) / (std_diff + mean_diff) if (std_diff + mean_diff) != 0 else 0
    
    time_total = timestamp[-1] - timestamp[0]
    time_event_density = len(timestamp) / time_total if time_total > 0 else 0
    
    time_diffs_prob = time_diff / np.sum(time_diff)  # Normalize
    time_entropy = -np.sum(time_diffs_prob * np.log2(time_diffs_prob + 1e-9))

    return {
        "time_diffmean": time_diffmean,
        "time_diffqmean": time_diffqmean,
        "time_diffmax": time_diffmax,
        "time_diffmin": time_diffmin,
        "time_diffmedian": time_diffmedian,

        'time_diffstd': time_diffstd,
        'time_diffvar': time_diffvar,
        'time_burstiness': time_burstiness,
        'time_total': time_total,
        'time_event_density': time_event_density,
        'time_entropy': time_entropy
    }

def signal_stat_feature_extraction(signal: np.ndarray) -> Dict[str, Any]:
    """
    Extract signal statistical feature
    """
    signal_diff = np.diff(signal)
    signal_diff = signal_diff[~np.isnan(signal_diff)]

    value_count = np.count_nonzero(~np.isnan(signal))
    value_median = np.median(signal)
    value_mean = signal.mean()
    value_qmean = np.mean([np.percentile(signal, [25, 75])])
    value_max = signal.max()
    value_min = signal.min()    
    value_maxmin = value_max - value_min
    value_std = signal.std() 
    value_var = signal.var() 
    
    value_diffmax = signal_diff.max()
    value_diffmin = signal_diff.min()
    value_diffmean  = signal_diff.mean()
    value_diffqmean = np.mean([np.percentile(signal_diff, [25, 75])])
    value_diffmedian = np.median(signal_diff)
    value_diffmaxmin = value_diffmax - value_diffmin
    value_diffstd = signal_diff.std() 
    value_diffvar = signal_diff.var()

    return {
        "value_count": value_count,
        "value_median": value_median,
        "value_mean": value_mean,
        "value_qmean": value_qmean,
        "value_max": value_max,
        "value_min": value_min,
        "value_maxmin": value_maxmin,
        "value_std": value_std,
        "value_var": value_var,
        "value_diffstd": value_diffstd,
        "value_diffvar": value_diffvar,
        "value_diffmax": value_diffmax,
        "value_diffmin": value_diffmin,
        "value_diffmean": value_diffmean,
        "value_diffqmean": value_diffqmean,
        "value_diffmedian": value_diffmedian,
        "value_diffmaxmin": value_diffmaxmin,
    }

def tsfel_feature_extraction(signal: np.ndarray, timestamp: np.ndarray) -> Dict[str, Any]:
    """
    Extract signal features with tsfel
    """
    # Get the default TSFEL configuration (features from all domains)
    cfg = tsfel.get_features_by_domain()
    
    ts = np.linspace(timestamp.min(), timestamp.max(), num=len(signal))
    features_df = tsfel.time_series_features_extractor(
        cfg, signal,
        fs=1/((ts[1]-ts[0])/3600),
        verbose=False
    )
    
    # TSFEL returns a DataFrame with one row per signal. Convert that row to a dictionary.
    # If signal is 1D, you typically get one row. We take .iloc[0] to get that row as a Series.
    features_dict = features_df.iloc[0].to_dict()
    return features_dict

def input_split(signal: np.ndarray, timestamp: np.ndarray, split_num: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Splits the signal and timestamp arrays into equal parts based on split_num.
    
    Args:
        signal (np.ndarray): The array containing signal data.
        timestamp (np.ndarray): The array containing timestamp data.
        split_num (int): The number of splits to create.
    
    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Two lists, each containing the split signal
                                                   and timestamp subarrays, respectively.
    """
    # Check that signal and timestamp have the same length
    if signal.shape[0] != timestamp.shape[0]:
        raise ValueError("signal and timestamp arrays must have the same length.")
    
    # Check if split_num is valid
    if split_num <= 0 or split_num > signal.shape[0]:
        raise ValueError("split_num must be a positive integer less than or equal to the length of the input arrays.")
    
    # Use np.array_split to split the signal and timestamp arrays
    split_signal = np.array_split(signal, split_num)
    split_timestamp = np.array_split(timestamp, split_num)
    
    # Convert to lists for the return type
    return list(split_signal), list(split_timestamp)

def feature_extraction(datapoint: dict, split_num: int):
    signal_list, timestamp_list = input_split(
        signal=datapoint['v'],
        timestamp=datapoint['t'].astype('timedelta64[s]').astype(int),
        split_num=split_num
    )

    feature_list = []
    for i in range(len(signal_list)):
        time_feat = timestamp_feature_extraction(timestamp_list[i])
        sig_feat = tsfel_feature_extraction(signal_list[i], timestamp_list[i])
        sig_stat_feat = signal_stat_feature_extraction(signal_list[i])
        feature_list.append({
            **sig_feat, **time_feat, **sig_stat_feat
        })

    return feature_list

def preprocessing(trn_x_path, trn_y_path, tst_x_path, split_num: int):
    train_y = pd.read_csv(trn_y_path)
    train_x = ZipFile(trn_x_path, 'r')

    feat_keys = None
    train_features = []
    for i in tqdm(range(len(train_y))):
        datapoint = pickle.loads(train_x.read('train_X/' + train_y.filename[i]))
        features = feature_extraction(datapoint, split_num)

        if i == 0:
            feat_keys = np.array(list(features[0].keys()))
        
        feat_mtx = np.stack([list(feat.values()) for feat in features], axis=0)
        train_features.append(feat_mtx)

    train_features = np.stack(train_features, axis=0)
    train_features = np.nan_to_num(train_features)

    test_x = ZipFile(tst_x_path, 'r')
    listtestfile = test_x.namelist()[1:]
    test_features = []
    for test_file in tqdm(listtestfile):
        datapoint = pickle.loads(test_x.read(test_file))
        features = feature_extraction(datapoint, split_num)
        feat_mtx = np.stack([list(feat.values()) for feat in features], axis=0)
        test_features.append(feat_mtx)
        
    test_features = np.stack(test_features, axis=0)
    test_features = np.nan_to_num(test_features)

    np.save("./train_features.npy", train_features)
    np.save("./test_features.npy", test_features)
    np.save("./feature_keys.npy", feat_keys)

if __name__ == '__main__':
    trn_y_path = "./downloads/train_y_v0.1.0.csv"
    trn_x_path = "./downloads/train_X_v0.1.0.zip"
    tst_x_path = "./downloads/test_X_v0.1.0.zip"

    preprocessing(
        trn_x_path,
        trn_y_path,
        tst_x_path,
        split_num=3
    )