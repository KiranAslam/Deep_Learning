import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split

class Iris_dataset:
    def __init__(self,test_size=0.20,random_state=42):
        self.test_size=test_size
        self.random_state=randon_state
        data=sklearn.datasets.load_iris()
        self.X=data.data
        self.y=data.target.reshape(-1,1)

    def normalize(self,X_train,X_test):
        mean=X_train.mean(axis=0)
        std=X_train.std(axis=0)+1e-8
        X_train=(X_train-mean)/std
        X_test=(X_test-mean)/std
        return X_train,X_test
    def one_hot_encoding(self,y):
        return np.eye(3)[y.flatten()]
    def get_data(self):
        X_train,X_test,y_train,y_test=train_test_split(
            self.X,self.y,test_size=self.test_size,random_state=self.random_state
        )
        y_train = self.one_hot_encoding(y_train)
        y_test  = self.one_hot_encoding(y_test)
        X_train,X_test=self.normalize(X_train,X_test)
        return X_train,X_test,y_train,y_test

