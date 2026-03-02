import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split

class Iris_dataset:
    def __init__(self,test_size=0.20,random_state=42):
        self.test_size=test_size
        self.random_state=random_state
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
class CategoricalCrossEntropy:
    
    @staticmethod
    def forward(self,y_true,y_pred):
        eps=1e-15
        y_pred=np.clip(y_pred,eps,1-eps)
        return -np.mean(np.sum(y_true*np.log(y_pred),axis=1))

class Neural_Network:
    def __init__(self,input_size=4,hidden_size=8,output_size=3):

        self.w1=np.random.randn(input_size,hidden_size)*0.01
        self.b1=np.zeros((1,hidden_size))

        self.w2=np.random.randn(hidden_size,output_size)*0.01
        self.b2=np.zeros((1,output_size))

        self.A1=None
        self.Z1=None
        self.A2=None
        self.Z2=None
    
    def forward(self,X):
        self.X=X
        self.Z1=np.dot(X,self.w1) + self.b1
        self.A1=ReLU.forward(self.Z1)

        self.Z2=np.dot(self.A1,self.w2) + self.b2
        self.A2=Softmax.forward(self.Z2)

        return self.A2
    def backward(self,y_true):
        m=self.X.shape[0]
        dZ2=self.A2-y_true
        dw2=(1/m)*np.dot(self.A1.T,dZ2)
        db2=(1/m)*np.sum(dZ2,axis=0,keepdims=True)
        dA1=np.dot(dZ2,self.w2.T)
        dZ1=dA1*(self.Z1>0)
        dw1=(1/m)*np.dot(self.X.T,dZ1)
        db1=(1/m)*np.sum(dZ1,axis=0,keepdims=True)
        self.dw2 = dw2
        self.db2 = db2
        self.dw1 = dw1
        self.db1 = db1
        
    def update(self,lr=0.01):
        self.w2 -= lr*self.dw2
        self.b2 -= lr*self.db2
        self.w1 -= lr*self.dw1
        self.b1 -= lr*self.db1

    def predict(self,X):
        probs=self.forward(X)
        return np.argmax(probs,axis=1)
class Trainer:
    def __init__(self,model,loss_fn,lr=0.01):
        self.model=model
        self.loss_fn=loss_fn
        self.lr=lr
        self.loss_history=[]
    def Train(self,X,y,epochs=2000):
        for epoch in range(epochs):
            y_pred=self.model.forward(X)
            loss=self.loss_fn.forward(y,y_pred)
            self.loss_history.append(loss)

            self.model.backward(y)
            self.backward.update(self.lr)
            if epoch%100 == 0:
                 print(f"Epoch: {epochs} |  Loss: {loss}")
            return self.loss_history

 
dataset = Iris_dataset()
X_train, X_test, y_train, y_test = dataset.get_data()

model = Neural_Network()
trainer=Trainer(model=model,loss_fn=CategoricalCrossEntropy,lr=0.1)
loss_history=trainer.Train(X_train,y_train,epochs=2000)
preds=model.predicts(X_test)
Accuracy=np.mean(preds==y_test)
print(f"Accuracy:{Accuracy*100:.2f}%")

plt.plot(loss_history)
plt.title("Multi Layer perceptron")
plt.xlabel("Epoches")
plt.ylabel("loss")
plt.show()
  

