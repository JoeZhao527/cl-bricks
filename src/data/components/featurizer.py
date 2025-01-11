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

