import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


class Sigmoid:
    @staticmethod  
    def forward(z):
        return 1/(1+np.exp(-z))
    @staticmethod
    def backward(z):
        s=Sigmoid.forward(z)
        return s*(1-s)

class Losses:
    @staticmethod
    def forward(y_true,y_pred):
        eps=1e-8
        return -np.mean(y_true*np.log(y_pred+eps)+(1-y_true)* np.log(1-y_pred+eps))
    @staticmethod
    def backward(y_true,y_pred):
        eps=1e-8
        return (y_pred - y_true) / ((y_pred + eps)* (1 - y_pred + eps))

class Neuron:
    def __init__(self,input_feature):
        self.w=np.random.rand(input_feature,1)*0.01
        self.b=np.zeros((1,))
        self.X = self.Z = self.A = None
        self.dw = self.db = None
 
    def forward(self,X):
        self.X=X
        self.Z=np.dot(X,self.w)+self.b
        self.A=Sigmoid.forward(self.Z)
        return self.A


    def backward(self,dA):
        m=self.X.shape[0]
        self.dZ=dA*Sigmoid.backward(self.Z)
        self.dw=(1/m)*np.dot(self.X.T,self.dZ)
        self.db=(1/m)*np.sum(self.dZ)
        return self.dZ

    def update(self,lr):
        self.w -=lr*self.dw
        self.b -=lr*self.db
class Trainer:
    def __init__(self,Model,loss_fn,lr=0.1):

        self.Model=Model
        self.loss_fn=loss_fn
        self.lr = lr
        self.history_loss=[]

    def Train(self,X,y,epochs=1000):
        for epoch in range(epochs):
            y_pred=self.Model.forward(X)
            loss=self.loss_fn.forward(y,y_pred)
            self.history_loss.append(loss)

            dA=self.loss_fn.backward(y,y_pred)
            self.Model.backward(dA)

            self.Model.update(self.lr)

            if epoch % 100 ==0:
                print(f"Epochs:{epoch} | Loss:{loss:.4f}")
        return self.history_loss

X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                           n_clusters_per_class=1, random_state=42)

y=y.reshape(-1,1)

Model=Neuron(input_feature=2)
loss_fn=Losses()
trainer=Trainer(Model,loss_fn,lr=0.1)
losses=trainer.Train(X,y,epochs=1000)

plt.plot(losses)
plt.title("Single Neuron training")
plt.xlabel("Epoches")
plt.ylabel("loss")
plt.show()
