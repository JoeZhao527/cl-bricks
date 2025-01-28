from xgboost import XGBClassifier
import numpy as np

class CustomXGBClassifier(XGBClassifier):
    def __init__(self, *, class_weight=None, **kwargs):
        super().__init__(**kwargs)

        self.class_weight = class_weight

    def init_classes(self, y):
        unique_classes = np.unique(y)
        class_to_int = {_cls: idx for idx, _cls in enumerate(unique_classes)}
        int_to_class = {idx: _cls for _cls, idx in class_to_int.items()}

        self.class_to_int = class_to_int
        self.int_to_class = int_to_class

    def fit(self, x, y, **kwargs):
        # Map labels to numeric values
        self.init_classes(y)
        y = np.array([self.class_to_int[label] for label in y])

        if self.class_weight:
            sample_weight = np.ones(y.shape[0], dtype = 'float')
            for i, _label in enumerate(y):
                sample_weight[i] = self.class_weight[self.int_to_class[_label]]
            
            super().fit(x, y, sample_weight=sample_weight, **kwargs)
        else:
            super().fit(x, y, **kwargs)

    def get_class_names(self):
        return [self.int_to_class[c] for c in self.classes_]