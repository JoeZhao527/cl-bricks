import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from ensemble.config.labels import LABEL_TIERS, LEVEL_LABLES
from sklearn.model_selection import StratifiedKFold


def get_active_labels(row):
    arr = row.to_numpy() # convert to numpy array
    indices = np.where(arr == 1)[0] # get indices where value is 1
    labels = row.index[indices].tolist() # get labels from indices
    return labels

class LabelTiers:
    def __init__(self, train_y_path: str):
        self.train_y = pd.read_csv(train_y_path)
        self.level_labels = self._get_level_labels()
        self.label_tiers = {i+1: self.level_labels[i] for i in range(LABEL_TIERS)}
        self.padded_labels = self._prepare_padded_labels()
        
        # Regression labels for counts of labels
        self.zeros_count = self.train_y.drop(columns=['filename']).eq(0).sum(axis=1)
        self.ones_count = self.train_y.drop(columns=['filename']).eq(1).sum(axis=1)

    def _get_level_labels(self, load: bool = True):
        if load:
            return LEVEL_LABLES
        
        level_labels = [list(self.train_y.columns[1:]), [], [], [], []]

        for k in range(0, 4):
            check_labels = level_labels[k]
            label_len = len(check_labels)
            idx_is_subset_of_col = pd.DataFrame(0, index=check_labels, columns=check_labels)
            is_subset = []

            for i in tqdm(range(label_len), desc=f"[{datetime.now()}] Processing level {k}"):
                for j in range(label_len):
                    src_lb, tgt_lb = check_labels[i], check_labels[j]
                    src = self.train_y[self.train_y[src_lb] == 1]
                    tgt = self.train_y[(self.train_y[src_lb] == 1) & (self.train_y[tgt_lb] == 1)]

                    idx_is_subset_of_col.loc[src_lb, tgt_lb] = len(src) <= len(tgt)
                    if len(src) <= len(tgt) and src_lb != tgt_lb:
                        is_subset.append([src_lb, tgt_lb])

            remove_label = set([s[0] for s in is_subset])
            
            for rl in remove_label:
                level_labels[k].remove(rl)
                level_labels[k+1].append(rl)

        return level_labels
    
    def get_tier(self, label):
        for tier_num, tier_list in self.label_tiers.items():
            if label in tier_list:
                return tier_num
        return None

    def sort_labels(self, labels):
        return sorted(labels, key=lambda label: (self.get_tier(label) or float('inf'), label))

    def _prepare_padded_labels(self):
        label_hier = self.train_y.apply(get_active_labels, axis=1).tolist()
        
        sorted_labelhier = [self.sort_labels(labels) for labels in label_hier]
        label_hier = np.array(
            sorted_labelhier,
            dtype=object,
        )

        # Pad sample with labels less than full `LABEL_TIERS` with `None`
        # E.g. ["Sensor", "Power_Sensor", "None", "None", "None"]
        padded_label = pd.Series(label_hier).apply(
            lambda x: x + ['None'] * (LABEL_TIERS - len(x))
            if len(x) < LABEL_TIERS else x
        )

        return padded_label
    
    def create_folds(self, n_splits=10, random_state=96):
        # Create a label array for stratification
        # We'll use the first non-zero label for each row as the stratification target
        stratify_labels = []
        for _, row in self.train_y.iterrows():
            labels = row[self.train_y.columns != 'filename'].values
            # Get first non-negative label, or 0 if all negative
            first_positive = next((i for i, x in enumerate(labels) if x >= 0), 0)
            stratify_labels.append(first_positive)
        
        # Create StratifiedKFold object
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Generate fold indices
        folds = []
        for train_idx, val_idx in skf.split(self.train_y, stratify_labels):
            folds.append({
                'train': train_idx,
                'val': val_idx
            })
        
        return folds
