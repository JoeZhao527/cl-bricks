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

from .models.xgboost import CustomXGBClassifier
from .models.standard_sklearn import CustomRandomForestClassifier, CustomLGBMClassifier, CustomCatBoostClassifier

model_zoo = {
    "random_forest": CustomRandomForestClassifier,
    "lightgbm": CustomLGBMClassifier,
    "xgboost": CustomXGBClassifier,
    "catboost": CustomCatBoostClassifier
}
