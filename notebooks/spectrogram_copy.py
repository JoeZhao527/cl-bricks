import pandas as pd
from tqdm import tqdm
import numpy as np
from pycaret.classification import *
from zipfile import ZipFile
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy import signal
import random
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
import math

# ------------------------
# Function Definitions
# ------------------------

def get_spec(file_name, zipf, interval=None):
    data = pickle.loads(zipf.read(file_name))

    # Extract time (t) and signal (x)
    t = data['t'].astype('timedelta64[s]').astype(int)  # Time in seconds
    x = data['v']  # Signal values

    # Handle duplicate timestamps
    unique_t, unique_indices = np.unique(t, return_index=True)
    unique_x = x[unique_indices]

    # Define the new time grid (300 seconds interval)
    start_time = unique_t.min()
    end_time = unique_t.max()
    new_t = np.arange(start_time, end_time + 1, 300)  # 300s interval

    # Interpolate the signal to the new time grid
    interpolation_function = interp1d(unique_t, unique_x, kind='cubic', fill_value="extrapolate")
    new_x = interpolation_function(new_t)

    x = new_x
    t = new_t

    if interval is not None:
        if len(x) < interval:
            # Avoid errors if the signal is shorter than the interval
            x = np.pad(x, (0, max(0, interval - len(x))), 'constant')
            t = np.pad(t, (0, max(0, interval - len(t))), 'constant')
        start_point = random.randint(0, max(0, len(x) - interval))
        x = x[start_point:start_point + interval]
        t = t[start_point:start_point + interval]

    # Compute the sampling interval (average time difference between samples)
    dt = np.mean(np.diff(t))  # Time difference in seconds

    # Compute the sampling frequency
    fs = 1.0 / dt  # Sampling frequency in Hz

    # Compute the spectrogram
    frequencies, times, Sxx = signal.spectrogram(
        x,
        fs=fs,
        window='hann',
        scaling='density',
        mode='magnitude'
    )

    return frequencies, times, Sxx

def process_batch(batch_filenames, zip_path, interval=None):
    """
    Processes a batch of files: opens the zip once and computes spectrograms for all files in the batch.

    Args:
        batch_filenames (list): List of filenames to process.
        zip_path (str): Path to the zip file.
        interval (int, optional): Interval parameter for get_spec. Defaults to None.

    Returns:
        dict: Mapping from filename to spectrogram (Sxx).
    """
    spec_dict = {}
    try:
        with ZipFile(zip_path, 'r') as zipf:
            for file_name in batch_filenames:
                _, _, sxx = get_spec(file_name, zipf, interval=interval)
                spec_dict[file_name] = sxx
    except Exception as e:
        print(f"Error processing batch: {e}")
    return spec_dict

def split_into_batches(lst, n_batches):
    """
    Splits a list into n_batches approximately equal parts.

    Args:
        lst (list): The list to split.
        n_batches (int): Number of batches.

    Yields:
        list: Next batch of the list.
    """
    batch_size = math.ceil(len(lst) / n_batches)
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

if __name__ == '__main__':
    # ------------------------
    # File Paths
    # ------------------------
    trn_y_path = "../downloads/train_y_v0.1.0.csv"
    trn_x_path = "../downloads/train_X_v0.1.0.zip"
    tst_x_path = "../downloads/test_X_v0.1.0.zip"

    # ------------------------
    # Load Labels
    # ------------------------
    train_y = pd.read_csv(trn_y_path)

    # Prepare filename lists
    filename_trn = list(train_y.filename.apply(lambda x: "train_X/" + x))
    
    # Open the test zip to get test filenames
    with ZipFile(tst_x_path, 'r') as zipf_test:
        filename_tst = zipf_test.namelist()[1:]  # Assuming the first entry is not needed

    # ------------------------
    # Configuration
    # ------------------------
    max_workers = 6  # Adjust based on your CPU cores
    interval = None  # Set if needed

    # ------------------------
    # Create Batches
    # ------------------------
    # Total number of batches is equal to max_workers
    trn_batches = list(split_into_batches(filename_trn, max_workers))
    tst_batches = list(split_into_batches(filename_tst, max_workers))

    # ------------------------
    # Initialize spec_feat
    # ------------------------
    spec_feat = {}

    # ------------------------
    # Process Train Files in Batches
    # ------------------------
    print("Processing Train Files...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        futures = [
            executor.submit(process_batch, batch, trn_x_path, interval)
            for batch in trn_batches
        ]

        # Collect results as they complete
        for future in tqdm(futures, total=len(futures), desc="Train Batches"):
            batch_result = future.result()
            spec_feat.update(batch_result)

    # ------------------------
    # Process Test Files in Batches
    # ------------------------
    print("Processing Test Files...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        futures = [
            executor.submit(process_batch, batch, tst_x_path, interval)
            for batch in tst_batches
        ]

        # Collect results as they complete
        for future in tqdm(futures, total=len(futures), desc="Test Batches"):
            batch_result = future.result()
            spec_feat.update(batch_result)

    # ------------------------
    # Save or Further Process spec_feat
    # ------------------------
    # Example: Save to a pickle file
    with open("spec_feat.pkl", "wb") as f:
        pickle.dump(spec_feat, f)

    print("Spectrogram computation completed and saved to spec_feat.pkl")
