from xgboost import XGBClassifier
import numpy as np

class CustomXGBClassifier(XGBClassifier):
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
        super().fit(x, y, **kwargs)

    def get_class_names(self):
        return [self.int_to_class[c] for c in self.classes_]