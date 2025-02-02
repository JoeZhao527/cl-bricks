from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

class CustomRandomForestClassifier(RandomForestClassifier):
    def get_class_names(self):
        return self.classes_
    
class CustomLGBMClassifier(LGBMClassifier):
    def get_class_names(self):
        return self.classes_