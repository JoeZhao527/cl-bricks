from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

from catboost import CatBoostClassifier

class CustomRandomForestClassifier(RandomForestClassifier):
    def get_class_names(self):
        return self.classes_
    
class CustomLGBMClassifier(LGBMClassifier):
    def get_class_names(self):
        return self.classes_
    
class CustomCatBoostClassifier(CatBoostClassifier):
    def get_class_names(self):
        return self.classes_