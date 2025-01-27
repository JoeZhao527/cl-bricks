import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import List


def bin_normalization(df_list: List[pd.DataFrame], bin_nums=1000):
    """
    Perform column-wise normalization for a list of dataframe, in a bin manner

    Args:
        df_list (list of pd.DataFrame): A list of DataFrames to normalize
        bin_nums (int, optional): The number of bins to create for each column. Defaults to 1000.

    Returns:
        list of pd.DataFrame: A list of DataFrames with the same size and original order as `df_list`,
                              but with the columns binned based on quantiles.
    """
    big_df = pd.concat(df_list, ignore_index=True)
    
    for col in tqdm(list(big_df.columns), desc=f"[{datetime.now()}] normalizing"):
        cut_bins = []
        score_labels = []
        quants = big_df[col].quantile([i / bin_nums for i in range(bin_nums)])
        quants[0] = -np.inf
        for i, quant in enumerate(quants):
            if len(cut_bins) == 0 or quant != cut_bins[-1]:
                cut_bins.append(quant)
                score_labels.append(i)

        cut_bins.append(np.inf)
        
        big_df[col] = pd.cut(
            big_df[col],
            bins=cut_bins,
            labels=score_labels
        ).astype(float)

    result_list = []
    start_idx = 0
    for df in df_list:
        size = len(df)
        sub_df = big_df.iloc[start_idx : start_idx + size].copy().reset_index(drop=True)
        result_list.append(sub_df)
        start_idx += size

    return result_list



def feature_crossing(df_list: List[pd.DataFrame], corr_thr: float = 0.2):
    # Concatenate all DataFrames
    big_df = pd.concat(df_list, ignore_index=True)

    # Select the pair of features that has low correlation
    corr_matrix = big_df.corr()

    # Create a list of (x, y, corr) from the correlation matrix
    corr_list = []
    for col in tqdm(list(corr_matrix.columns), desc=f"[{datetime.now()}] Comupting pairwise feature correlation"):
        for row in corr_matrix.index:
            if col != row:  # Exclude the diagonal values (self-correlation)
                corr_list.append((row, col, corr_matrix.at[row, col]))

    corr_df = pd.DataFrame(corr_list, columns=['x', 'y', 'corr'])
    
    non_corr = corr_df[(corr_df['corr'] < corr_thr) & (corr_df['corr'] > -corr_thr)].drop_duplicates('corr')
    
    # Currently only crossing features within 
    non_corr = non_corr[
        (non_corr.x.str.contains("0_") & non_corr.y.str.contains("0_")) |
        (non_corr.x.str.contains("time_") & non_corr.y.str.contains("time_")) |
        (non_corr.x.str.contains("value_") & non_corr.y.str.contains("value_"))
    ]
    non_corr = non_corr.drop_duplicates('y').drop_duplicates('x')
    non_corr['combine'] = non_corr.apply(lambda x: f"{x.x}_{x.y}", axis=1)
    combine_info = non_corr.to_dict("records")

    # feature crossing by adding the selected pair of features
    for row in combine_info:
        big_df[row['combine']] = (big_df[row['x']] + big_df[row['y']]) / 2

    # Split back into the original list of DataFrames
    result_list = []
    start_idx = 0
    for df in df_list:
        size = len(df)
        sub_df = big_df.iloc[start_idx : start_idx + size].copy().reset_index(drop=True)
        result_list.append(sub_df)
        start_idx += size

    return result_list

def train_test_wrapper(train: List[pd.DataFrame], test: pd.DataFrame, func):
    # Helper function for easier preprocessing
    res_list = func(train + [test])
    return res_list[:-1], res_list[-1]

def train_test_list_wrapper(train: List[pd.DataFrame], test: List[pd.DataFrame], func):
    res_list = func(train + test)
    return res_list[:len(train)], res_list[len(train):]
