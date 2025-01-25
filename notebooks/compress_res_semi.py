import pandas as pd
import numpy as np

dev = pd.read_csv("0124_semi_supervise_balanced_thr090.csv")
arr = dev.drop(columns=["filename"]).values
np.save("0124_semi_supervise_balanced_thr090.npy", np.stack(np.where(arr ==1)))