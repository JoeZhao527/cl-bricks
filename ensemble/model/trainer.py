from typing import List
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle

from sklearn.metrics import precision_recall_fscore_support

from ensemble.config.labels import LABEL_TIERS, LABEL_NAMES
from ensemble.model.predictor import get_test_agg, get_stacked_res, post_processing


def get_data(train_X: List[pd.DataFrame], _label: np.array, none_ratio_thr: float, fold, fold_idx):
    # Prepare train and validation data for this fold
    train_X_fold_list = []
    train_y_fold_list = []
    for trn_x in train_X:
        train_X_fold_list.append(trn_x.iloc[fold['train']])
        train_y_fold_list.append(_label[fold['train']])

    train_X_fold = pd.concat(train_X_fold_list)
    train_y_fold = np.concatenate(train_y_fold_list)

    valid_X_fold_list = []
    valid_y_fold_list = []
    for trn_x in train_X:
        valid_X_fold_list.append(trn_x.iloc[fold['val']])
        valid_y_fold_list.append(_label[fold['val']])

    val_X_fold = pd.concat(valid_X_fold_list)
    val_y_fold = np.concatenate(valid_y_fold_list)
    
    # Check the train_y_fold. If more than 30% of samples are labeled "None",
    # randomly sample from the "None" to make that ratio no more than 30%.
    none_mask = (train_y_fold == "None")
    none_count = np.sum(none_mask)
    total_samples = len(train_y_fold)
    none_ratio = none_count / total_samples if total_samples > 0 else 0

    if none_ratio > none_ratio_thr:
        # Calculate how many "None" labels we should keep (30% of total)
        max_none_to_keep = int(none_ratio_thr * (total_samples - none_count))

        # Randomly choose which "None" labels to keep
        none_indices = np.where(none_mask)[0]

        # Fix the random seed before shuffling for reproducibility
        rng = np.random.RandomState(fold_idx)
        rng.shuffle(none_indices)
        
        keep_none_indices = none_indices[:max_none_to_keep]

        # Indices of all non-"None" labels
        other_indices = np.where(~none_mask)[0]

        # Combine indices to keep and then sort
        new_indices = np.concatenate([keep_none_indices, other_indices])
        new_indices = np.sort(new_indices)  # Sort so we can index the DataFrame consistently

        # Subset the training data
        train_X_fold = train_X_fold.iloc[new_indices]
        train_y_fold = train_y_fold[new_indices]

    return train_X_fold, train_y_fold, val_X_fold, val_y_fold


def prepare_val_prediction(train_X: List[pd.DataFrame], fold, val_X_fold: pd.DataFrame, val_pred_df: pd.DataFrame):
    """
    Prepare the prediction result of validation set for downstream analysis

    Args:
        train_X: a list of training data, for marking validation dataset index
        fold: fold split that contains data id for each sample
        val_X_fold: validation fold input feature df
        val_pred_df: validation fold prediction results
    """
    val_fold_info = []
    for _f in range(len(train_X)):
        f_info = pd.DataFrame(data=fold['val'], columns=["fold_idx"])
        f_info['dataset_idx'] = _f
        val_fold_info.append(f_info)

    val_fold_idx = pd.concat(val_fold_info)
    
    val_feat_df = pd.concat([
        val_fold_idx.reset_index(drop=True),
        val_X_fold.reset_index(drop=True),
        val_pred_df,
    ], axis=1)

    return val_feat_df


def train_base_model(
    train_X: List[pd.DataFrame],
    _label,
    folds,
    model_class,
    params: dict,
    none_ratio_thr: float,
    level_id: int,
):
    """
    Train base models using k-fold cross validation
    
    Args:
        train_X: Training features DataFrame
        _label: Array of labels
        folds: List of dictionaries containing train/val indices
        
    Returns:
        tuple: (list of trained classifiers, list of scores, list of validation predictions)
    """
    classifiers = []
    val_feat_df_list = []

    for f_idx, fold in enumerate(folds):
        train_X_fold, train_y_fold, val_X_fold, _ = get_data(
            train_X=train_X,
            _label=_label,
            none_ratio_thr=none_ratio_thr,
            fold=fold,
            fold_idx=f_idx
        )

        print(f"Train size: {len(train_X_fold)}, Valid size: {len(val_X_fold)}")
        
        # Create and train Random Forest model
        model = model_class(**params)
        model.fit(train_X_fold, train_y_fold)
        
        classifiers.append(model)
        
        # Calculate score and save predictions on validation set
        val_preds = model.predict_proba(val_X_fold)
        val_pred_df = pd.DataFrame(data=val_preds, columns=model.get_class_names())
        val_pred_df = val_pred_df.rename(columns={col: f"{col}_{level_id}" for col in list(model.get_class_names())})

        val_feat_df = prepare_val_prediction(
            train_X=train_X,
            fold=fold,
            val_X_fold=val_X_fold,
            val_pred_df=val_pred_df
        )

        val_feat_df_list.append(val_feat_df)

    return classifiers, val_feat_df_list


def setup_prev_level_prediction(predictions, fold_num, num_datasets):
    new_train_level_x = pd.concat([predictions[i] for i in range(fold_num)]).sort_values(['dataset_idx', 'fold_idx'])
    return [
        new_train_level_x[new_train_level_x['dataset_idx'] == i] \
            .drop(columns=['dataset_idx', 'fold_idx']) \
            .reset_index(drop=True)
        for i in range(num_datasets)
    ]


def evaluate(label_df, pred_df):
    report = []
    for col in label_df:
        if col == "filename": continue

        col_eval = pd.DataFrame({"label": label_df[col], "pred": pred_df[col]})
        col_eval = col_eval[col_eval["label"] != 0]
        col_eval['label'] = (col_eval['label'] > 0).astype(int)
        col_eval['pred'] = (col_eval['pred'] > 0.5).astype(int)

        # Compute precision, recall, and f1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            col_eval["label"],
            col_eval["pred"],
            average="binary",
            zero_division=0  # to handle divisions by zero if any
        )
        
        # Add results to the report list
        report.append({
            "col": col,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        })
    
    # Convert the report list to a DataFrame
    report_df = pd.DataFrame(report)
    
    # Calculate averages (mean) for precision, recall, f1, and sum for support
    avg_row = {
        "col": "AVERAGE",
        "precision": report_df["precision"].mean(),
        "recall": report_df["recall"].mean(),
        "f1": report_df["f1"].mean(),
        "support": report_df["support"].sum()
    }

    print(f"[{datetime.now()}] Avg Precision: {avg_row['precision']:.3f}, Recall {avg_row['recall']:.3f}, F1 {avg_row['f1']}")
    report_df = pd.concat([report_df, pd.DataFrame([avg_row])], ignore_index=True)

    return report_df


class BaseModel:
    def __init__(
        self,
        model_cls,
        model_params,
        none_ratio_thr_list: List[float],
        train_input: List[pd.DataFrame],
        padded_labels,
        folds: List[dict],
        model_save_dir: str = None,
    ):
        self.train_input = train_input
        self.raw_feat_num = len(train_input[0].columns)
        self.model_cls = model_cls
        self.model_params = model_params
        self.none_ratio_thr_list = none_ratio_thr_list
        self.padded_labels = padded_labels
        self.folds = folds

        # Contains a list of list
        # After training, classifiers[i] will contain the models of each fold in tier_`i`
        # E.g. for 10 folds cross validation, its shape will be (5, 10), where the first dimension is 
        # tier and the second dimension is folds
        self.classifiers = []

        # Same structure as the `self.classifiers`
        self.val_predictions = []

        # Whether to save model or keep it in the memory
        # Will save model into model_save_dir if provided. This is suitable for low memory machine.
        self.model_save_dir = model_save_dir
        if model_save_dir != None:
            os.makedirs(model_save_dir)
            print(f"[{datetime.now()}] Initialized model saving directory: {model_save_dir}")

    def load_ckpt(self, model_base_dir: str):
        self.classifiers = []

        for tier in range(LABEL_TIERS):
            tier_model_paths = []
            for fold in range(len(self.folds)):
                tier_model_paths.append(os.path.join(model_base_dir, f"tier_{tier}_fold_{fold}.pkl"))

            self.classifiers.append(tier_model_paths)

        print(f"[{datetime.now()}] models paths initialization completed")

    def train(self):
        # Train one model for each tier
        # the sub-tier model use super-tier model's prediction as input feature
        for i in range(LABEL_TIERS):
            print(f"Training level {i}")
            _classifiers, _val_predictions = train_base_model(
                self.train_input,
                np.array([x[i] for x in self.padded_labels]),
                self.folds,
                params=self.model_params,
                model_class=self.model_cls,
                none_ratio_thr=self.none_ratio_thr_list[i],
                level_id=i
            )

            # dump the models and save paths only if model_save_dir is provided
            if self.model_save_dir:
                _classifiers_paths = []
                for fid, clf in enumerate(_classifiers):
                    _clf_path = os.path.join(self.model_save_dir, f"tier_{i}_fold_{fid}.pkl")
                    with open(_clf_path, "wb") as f:
                        pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)

                    print(f"[{datetime.now()}] level {i} fold model {fid} to saved to {_clf_path}")

                _classifiers_paths.append(_clf_path)
                _classifiers = _classifiers_paths

            self.classifiers.append(_classifiers)
            self.val_predictions.append(_val_predictions)

            self.train_input = setup_prev_level_prediction(
                _val_predictions,
                fold_num=len(self.folds),
                num_datasets=len(self.train_input)
            )

    def get_validation_preds(self):
        # The last level validation prediction has the prediction for all labels
        # Concat all folds, where they are partition of the all the train data
        val_pred_res = pd.concat(self.val_predictions[-1])

        # Sort the results with dataset_idx and fold_idx for next level training
        val_pred_res = val_pred_res.sort_values(['dataset_idx', 'fold_idx']).reset_index(drop=True)

        return val_pred_res

    def evaluation(self, train_y: pd.DataFrame):
        """
        Macro precision, recall and f1 evaluation
        """
        # Get prediction results on validation set
        val_preds = self.get_validation_preds()
        
        # Get columns to split tiers
        label_start_col_idx = self.raw_feat_num + 2
        col_tiers = [[] for _ in range(LABEL_TIERS)]

        for col in list(val_preds.columns)[label_start_col_idx:]:
            col_tiers[int(col[-1])].append(col)

        # keep the label related columns
        # one more column for dataset_idx, one more for fold_idx
        val_preds = [
            g_df.sort_values("fold_idx").iloc[:, label_start_col_idx:]
            for _, g_df in val_preds.groupby("dataset_idx")
        ]

        # Prepare a list of list
        # The first level is tiers, and the second level is prediction from each dataset
        val_preds_list = []
        for cols in col_tiers:
            tier_preds = []
            for dataset_preds in val_preds:
                _df = dataset_preds[cols]
                tier_preds.append(_df.rename(columns={col: col[:-2] for col in _df.columns}).reset_index(drop=True))

            val_preds_list.append(tier_preds)

        # Use the same pipeline as test prediction to prepare final result
        val_level_pred_list = get_test_agg(val_preds_list)
        stacked_val_res = get_stacked_res(val_level_pred_list)
        val_final_res = post_processing(stacked_val_res, LABEL_NAMES, list(train_y['filename']))

        report = evaluate(train_y, val_final_res)

        return report