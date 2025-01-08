import pandas as pd
import numpy as np
import pickle
import tsfel
import json
import os
from typing import Tuple, List, Dict, Any
from tqdm import tqdm
from zipfile import ZipFile
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import traceback
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

def tsfel_feature_extraction(signal: np.ndarray, timestamp: np.ndarray, tsfel_freq_cfg: dict) -> Dict[str, Any]:
    """
    Extract signal features with tsfel
    """
    # interpolate values
    dt = 4838397.067/85922
    ts1 = np.linspace(timestamp.min(), timestamp.max(), num=len(signal))
    ts2 = np.arange(timestamp.min(), timestamp.max(), dt)
    # interpolator = interp1d(timestamp, signal, kind='nearest')
    # values_fixed = interpolator(ts1)
    # values_forfreq = interpolator(ts2)

    # statistical and temporal domain
    cfg1 = tsfel.get_features_by_domain(domain=['statistical', 'temporal'])
    features_df_1 = tsfel.time_series_features_extractor(
        cfg1, signal,
        fs=1/((ts1[1]-ts1[0])/3600),
        verbose=False
    )

    # frequency domain
    cfg2 = tsfel_freq_cfg
    features_df_2 = tsfel.time_series_features_extractor(
        cfg2, signal,
        fs=1/((ts2[1]-ts2[0])/3600),
        verbose=False
    )
    
    # TSFEL returns a DataFrame with one row per signal. Convert that row to a dictionary.
    # If signal is 1D, you typically get one row. We take .iloc[0] to get that row as a Series.
    features_dict = {
        **features_df_1.iloc[0].to_dict(),
        **features_df_2.iloc[0].to_dict()
    }
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

def feature_extraction(datapoint: dict, split_num: int, tsfel_freq_cfg: dict):
    signal_list, timestamp_list = input_split(
        signal=datapoint['v'],
        timestamp=datapoint['t'].astype('timedelta64[s]').astype(int),
        split_num=split_num
    )

    feature_list = []
    for i in range(len(signal_list)):
        time_feat = timestamp_feature_extraction(timestamp_list[i])
        sig_feat = tsfel_feature_extraction(signal_list[i], timestamp_list[i], tsfel_freq_cfg)
        sig_stat_feat = signal_stat_feature_extraction(signal_list[i])
        feature_list.append({
            **sig_feat, **time_feat, **sig_stat_feat
        })

    return feature_list

def process_datapoint(args):
    """
    Worker function to process a single datapoint.
    
    Args:
        args (tuple): Contains (zip_path, filename, split_num)
    
    Returns:
        Tuple[int, np.ndarray]: Index and the feature matrix for the datapoint
    """
    zip_path, filename, split_num, index, filename_prefix, tsfel_freq_cfg = args
    try:
        with ZipFile(zip_path, 'r') as zip_file:
            datapoint = pickle.loads(zip_file.read(filename_prefix + filename))
        features = feature_extraction(datapoint, split_num, tsfel_freq_cfg)
        feat_mtx = np.stack([list(feat.values()) for feat in features], axis=0)
        feat_mtx = np.nan_to_num(feat_mtx)
        return (index, feat_mtx)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return (index, None)

def preprocess_data(zip_path, filenames, split_num, feature_keys, tsfel_freq_cfg, num_workers=4, filename_prefix=""):
    """
    Preprocess data using multiprocessing.
    
    Args:
        zip_path (str): Path to the zip file.
        filenames (List[str]): List of filenames to process.
        split_num (int): Number of splits.
        feature_keys (List[str]): List of feature keys
        num_workers (int): Number of worker processes.
        filename_prefix (str): 
    
    Returns:
        np.ndarray: Feature matrix.
        np.ndarray: Feature keys.
    """
    feature_list = [None] * len(filenames)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Prepare arguments with indices to maintain order
        args = [(zip_path, filename, split_num, idx, filename_prefix, tsfel_freq_cfg) for idx, filename in enumerate(filenames)]
        # Use tqdm for progress bar
        futures = {executor.submit(process_datapoint, arg): arg[3] for arg in args}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {os.path.basename(zip_path)}"):
            idx, feat_mtx = future.result()
            if feat_mtx is not None:
                feature_list[idx] = feat_mtx
            else:
                # Handle failed datapoint, e.g., fill with zeros or remove
                feature_list[idx] = np.zeros((split_num, len(feature_keys)))
    
    # Convert list to numpy array
    feature_matrix = np.stack(feature_list, axis=0)
    return feature_matrix

def preprocessing(trn_x_path, trn_y_path, tst_x_path, split_num: int, output_dir: str, num_workers: int = 4):
    # Load training labels
    train_y = pd.read_csv(trn_y_path)
    
    # Get training filenames
    with ZipFile(trn_x_path, 'r') as train_zip:
        train_filenames = train_y['filename'].tolist()

    # initialize output directory
    os.makedirs(output_dir, exist_ok=False)


    with open("./data_prep/tsfel_freq_config.json", "r") as f:
        tsfel_freq_cfg = json.load(f)

    # Get feature keys from tsfel (assuming all datapoints have the same features)
    with ZipFile(trn_x_path, 'r') as train_zip:
        first_datapoint = pickle.loads(train_zip.read('train_X/' + train_filenames[0]))
    first_features = feature_extraction(first_datapoint, split_num, tsfel_freq_cfg)
    feat_keys = np.array(list(first_features[0].keys()))
    print(f"Feature dimensions: {len(feat_keys)}")
    np.save(os.path.join(output_dir, "feature_keys.npy"), feat_keys)
    
    # Preprocess training data
    train_features = preprocess_data(
        zip_path=trn_x_path,
        filenames=train_filenames[:2000],
        split_num=split_num,
        feature_keys=list(feat_keys),
        tsfel_freq_cfg=tsfel_freq_cfg,
        num_workers=num_workers,
        filename_prefix="train_X/"
    )
    
    # Save training features
    np.save(os.path.join(output_dir, "train_features.npy"), train_features)
    
    # Preprocess testing data
    with ZipFile(tst_x_path, 'r') as test_zip:
        test_filenames = test_zip.namelist()[1:]  # Assuming first file is not a data file
    
    test_features = preprocess_data(
        zip_path=tst_x_path,
        filenames=test_filenames[:2000],
        split_num=split_num,
        feature_keys=list(feat_keys),
        tsfel_freq_cfg=tsfel_freq_cfg,
        num_workers=num_workers
    )
    
    # Save testing features
    np.save(os.path.join(output_dir, "test_features.npy"), test_features)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess time-series data with feature extraction.'
    )
    
    parser.add_argument(
        '--trn_y_path',
        type=str,
        default="./downloads/train_y_v0.1.0.csv",
        help='Path to the training labels CSV file (e.g., ./downloads/train_y_v0.1.0.csv).'
    )
    
    parser.add_argument(
        '--trn_x_path',
        type=str,
        default="./downloads/train_X_v0.1.0.zip",
        help='Path to the training data ZIP file (e.g., ./downloads/train_X_v0.1.0.zip).'
    )
    
    parser.add_argument(
        '--tst_x_path',
        type=str,
        default="./downloads/test_X_v0.1.0.zip",
        help='Path to the testing data ZIP file (e.g., ./downloads/test_X_v0.1.0.zip).'
    )
    
    parser.add_argument(
        '--split_num',
        type=int,
        default=3,
        help='Number of splits for feature extraction (default: 3).'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of worker processes for multiprocessing (default: 8).'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default="./features",
        help='Path to the output dir (default: ./features)'
    )
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    
    preprocessing(
        trn_x_path=args.trn_x_path,
        trn_y_path=args.trn_y_path,
        tst_x_path=args.tst_x_path,
        split_num=args.split_num,
        output_dir=args.output_dir,
        num_workers=args.num_workers
    )