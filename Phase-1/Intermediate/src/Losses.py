import numpy as np

class BinaryCrossEntropy:
    @staticmethod
    def forward(y_true,y_pred):
        eps=1e-8
        return -np.mean(y_true * np.log(y_pred+eps)+ (1-y_true) *np.log(1-y_pred+eps))
    @staticmethod
    def backward(y_true,y_pred):
        eps=1e-8
        return (y_pred-y_true)/((y_pred+eps)*(1-y_pred+eps))