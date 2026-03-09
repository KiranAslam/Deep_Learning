import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split

class MNIST_Dataset:
    def __init__(self,random_state=42):
        self.random_state=random_state
        data=sklearn.datasets.fetch_openml('mnist_784',version=1)
        self.X=data.data
        self.y=data.target.astype(np.float32)
 
    def normalize(self,X_train,X_test,X_val):
        mean=X_train.mean(axis=0)
        std=X_train.std(axis=0)+1e-8
        X_train=(X_train-mean)/std
        X_test=X_test-mean/std
        X_val=X_val-mean/std
        return X_train,X_test,X_val

    def one_hot_encoding(self,y):
        num_class=10
        one_hot=np.zeros((len(y),num_class))
        one_hot[np.arrange(len(y),y)]=1
        return one_hot
    def train_data(self):
        return self.X_train,self.y_train
    def vald_data(self):
        return self.X_val,self.y_val
    def test_data(self):
        return self.X_test,self.y_test
    def get_data(self):
        #split : train+test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y,
            test_size=0.2,
            random_state=self.random_state
        )
        #split: train + validate
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=0.1,
            random_state=self.random_state
        )
        #Normalization
        X_train,X_test,X_val=normalize(
            self.X_train,self.X_test,self.X_val
        )
        #one hot encoding
        y_train=self.one_hot_encoding(self.y_train)
        y_test=self.one_hot_encoding(self.y_test)
        y_val=self.one_hot_encoding(self.y_val)

class ReLU:
    @staticmethod
    def forward(z):
        return np.maximum(0,z)
class Softmax:
    @staticmethod
    def forward(z):
        exp_values=np.exp(z-np.max(z,axis=1,keepdims=True))
        probabilities=exp_values/np.sum(exp_values,axis=1,keepdims=True)
        return probabilities
class CrossEntropyLoss:
    @staticmethod
    def forward(y_true,y_pred):
        m=y_true.shape[0]
        y_pred=np.clip(y_pred,1e-9,1-1e-9)
        return -np.mean(np.sum(y_true*np.log(y_pred),axis=1))

class DeepNeuralNetwork:
    def __init__(self,input_size=784,hidden_size1=128,hidden_size2=64,output_size=10):

        self.w1=np.random.randn(input_size,hidden_size1)*0.01
        self.b1=np.zeros((1,hidden_size1))

        self.w2=np.random.randn(hidden_size1,hidden_size2)*0.01
        self.b2=np.zeros((1,hidden_size2))

        self.w3=np.random.randn(hidden_size2,output_size)*0.01
        self.b3=np.zeros((1,output_size))

        self.A1=A1
        self.Z1=Z1
        self.A2=A2
        self.Z2=Z2
        self.A3=A3
        self.Z3=Z3

    def forward(self,X):
        self.X=X
        self.Z1=np.dot(X,self.w1)+self.b1
        self.A1=ReLU.forward(Z1)

        self.Z2=np.dot(self.A1,self.w2)+self.b2
        self.A2=ReLU.forward(Z2)

        self.Z3=np.dot(self.A2,self.w3)+self.b3
        self.A3=Softmax.forward(Z3)

        return self.A3

    def backward(self,y_true):
        m=self.X.shape[0]
        dZ3=self.A3-y_true
        dw3=(1/m)*np.dot(self.A2.T,dZ3)
        db3=(1/m)*np.sum(dZ3,axis=0,keepdims=True)

        dA2=np.dot(dZ3,self.w3.T)
        dZ2=dA2*(self.Z2>0)
        dw2=(1/m)*np.dot(self.A1.T,dZ2)
        db2=(1/m)*np.sum(dZ2,axis=0,keepdims=True)

        dA1=np.dot(dZ2,self.w2.T)
        dZ1=dA1*(self.Z1>0)
        dw1=(1/m)*np.dot(x.T,dZ1)
        db1=(1/m)*np.sum(dZ1,axis=0,keepdims=True)
        self.dw3 = dw3
        self.db3 = db3
        self.dw2 = dw2
        self.db2 = db2
        self.dw1 = dw1
        self.db1 = db1
    def update(self,lr=0.01):
        self.w3 -= lr*dw3
        self.b3 -= lr*db3
        self.w2 -= lr*db2
        self.b2 -= lr*db2
        self.w1 -= lr*dw1
        self.b1 -= lr*db1

    def predict(self,X):
        probs=self.forward(X)
        return np.argmax(probs,axis=1)





        


        

