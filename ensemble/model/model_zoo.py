from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier, XGBRegressor

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

from .models.xgboost import CustomXGBClassifier
from .models.standard_sklearn import CustomRandomForestClassifier, CustomLGBMClassifier

model_zoo = {
    "random_forest": CustomRandomForestClassifier,
    "lightgbm": CustomLGBMClassifier,
    "xgboost": CustomXGBClassifier,
    "rf_regressor": RandomForestRegressor,
    "xgb_regressor": XGBRegressor,
    "lgb_regressor": LGBMClassifier
}
