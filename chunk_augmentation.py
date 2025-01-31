import pandas as pd
from tqdm import tqdm
import numpy as np
from zipfile import ZipFile
import zipfile
import pickle
import os
import warnings
from ensemble.config.paths import PATHS
warnings.filterwarnings("ignore")

# Set the random seed for reproducibility
np.random.seed(42)

train_y = pd.read_csv(PATHS.train_y_path)
zipf_raw_train = ZipFile(PATHS.train_zip_path, 'r')

def process_and_shuffle_chunk(i, n_chunks):
    # Load the pickled data
    l_pkl = pickle.loads(zipf_raw_train.read(f'train_X/{train_y.filename[i]}'))
    
    # Create a DataFrame with timestamp and value
    data = pd.DataFrame({
        'timestamp': l_pkl['t'],  # Timestamp remains unchanged
        'value': l_pkl['v']
    })
    
    # Split 'value' into n_chunks parts
    value_chunks = np.array_split(data['value'], n_chunks)
    
    # Shuffle the chunks
    np.random.shuffle(value_chunks)
    
    # Concatenate the shuffled chunks back into a single series
    shuffled_values = pd.concat(value_chunks, ignore_index=True)
    
    # Recreate the DataFrame with shuffled values
    data['value'] = shuffled_values

    # Convert DataFrame back into the original pickle format
    l_pkl['v'] = data['value'].values  # Replace the original values with shuffled ones
    return l_pkl

if __name__ == '__main__':
    shuffle_chunks = [6, 7, 8, 9, 10]
    shuffle_output_base = "./processed/chunk_shuffle"
    os.makedirs(shuffle_output_base)

    for c_num in shuffle_chunks:
        output_path = os.path.join(shuffle_output_base, f"train_X_shuffle_{c_num}.zip")
        # To process all files and write back to the zip
        with zipfile.ZipFile(output_path, 'a') as zipf:  # Open the zip file in append mode
            for idx in tqdm(range(len(train_y.filename)), desc=f"Shuffling with n_chunks={c_num}"):  # Adjust to your dataset
                shuffled_data = process_and_shuffle_chunk(idx, n_chunks=c_num)
                
                # Pickle and write the processed data back to the zip file
                with zipf.open(f'train_X/{train_y.filename[idx]}', 'w') as f:
                    pickle.dump(shuffled_data, f)