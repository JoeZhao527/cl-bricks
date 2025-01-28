import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def merge_mean(df_list):
    """
    Given a list of DataFrames with identical columns and index,
    return one DataFrame whose numeric columns are the mean of all
    those numeric columns across the list.

    The "filename" column is assumed to be the same or can be taken
    from the first DataFrame.
    """
    # Make a copy of the first DataFrame to initialize the output
    merged_df = df_list[0].copy()

    # Identify all numeric columns (excluding "filename")
    numeric_cols = merged_df.columns.drop("filename")

    # Sum up the numeric columns from all other DataFrames
    for df in tqdm(df_list[1:], desc="Merging"):
        merged_df[numeric_cols] += df[numeric_cols]

    # Divide by the number of DataFrames to get the mean
    merged_df[numeric_cols] /= len(df_list)

    return merged_df

def check_pred_num(_final_res, thr=0.4):
    # Exclude 'filename' column if it exists
    filtered_df = _final_res.drop(columns=['filename'], errors='ignore')

    return (filtered_df >= thr).sum(axis=1)

if __name__ == '__main__':
    # base_dir = "./logs/ensemble/base_ensemble/01_27_2025-16_33_48"
    base_dir = "./logs/ensemble/base_ensemble/01_27_2025-21_34_30"
    thr = 0.5

    avg = [
        pd.read_csv(os.path.join(base_dir, "lgb/test_predictions", f"final_result_{i}.csv"))
        for i in tqdm(range(6), desc="Loading")
    ]

    m_res = merge_mean(avg)

    print(check_pred_num(m_res, thr).value_counts())

    for col in tqdm(list(m_res.columns)[1:], desc="Filtering"):
        m_res[col] = (m_res[col] > thr).astype(int)

    arr = m_res.drop(columns=["filename"]).values
    np.save("0127_lgb_recall.npy", np.stack(np.where(arr == 1)))