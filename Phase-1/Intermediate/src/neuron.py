import numpy as np
import random
from activations import sigmoid

class neuron:
    def __init__(self,input_features):
        self.w = np.random.randn(input_features,1)*0.01
        self.b = np.zeros((1,))

        self.X=None
        self.Z=None
        self.A=None
        self.dw=None
        self.db=None
    def forward(self,X):
        self.X=X
        self.Z=np.dot(X,self.w)+self.b
        self.A=sigmoid.forward(self.Z)
        return self.A
    def backward(self,dA):
        m=self.X.shape[0]
        dZ=dA*sigmoid.backward(self.Z)
        self.dw=(1/m)*np.dot(self.X.T,dZ)
        self.db=(1/m)*np.sum(dZ)
        return dZ
    def update(self,lr):
        self.w -= lr*self.dw
        self.b -=lr*self.db

    