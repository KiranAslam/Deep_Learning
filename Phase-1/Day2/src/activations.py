import numpy as np

class sigmoid:
    @staticmethod
    def forward(z):
        return 1/(1 + np.exp(-z))
    @staticmethod
    def backward(z):
        s=sigmoid.forward(z)
        return s*(1-s)