import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB


class ClassificationModel:
    def __init__(self):
        self._model = None
        
    @property
    def features(self):
        return self._features
    
    
    def train(self, df):
        X = df[self._features]
        y = df["성별"]

        self._model.fit(X, y)
    
    
    def predict(self, df):
        X = df[self._features]
        pred = self._model.predict(X)
        return pred
        

    

class Model01(ClassificationModel):
    def __init__(self):
        self._model = GaussianNB()
    
        self._features = [
            "눈높이",
            "목뒤높이",
            "손목둘레"
        ]
        
    # def train(self, df):
    #   You can override this function...
    
    
    # def predict(self, df):
    #   You can override this function...    
    
    
    
class Model02(ClassificationModel):
    def __init__(self):
        self._model = GaussianNB()
    
        self._features = [
            "눈높이",
            "목뒤높이",
            "손목둘레"
        ]
        
    # def train(self, df):
    #   You can override this function...
    
    
    # def predict(self, df):
    #   You can override this function...    
    
    
    

class Model03(ClassificationModel):
    def __init__(self):
        self._model = GaussianNB()
    
        self._features = [
            "눈높이",
            "목뒤높이",
            "손목둘레"
        ]
        
    # def train(self, df):
    #   You can override this function...
    
    
    # def predict(self, df):
    #   You can override this function...    
    
    