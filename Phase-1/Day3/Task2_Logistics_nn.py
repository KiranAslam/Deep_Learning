import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
# Loading breast cancer dataset
class BreastCancerDataset:
    def __init__(self,test_size=0.2,random_state=42):
        self.test_size=test_size
        self.random_state=random_state
        data = sklearn.datasets.load_breast_cancer()
        self.X = data.data
        self.y = data.target.reshape(-1, 1)
        
    def normalize(self,X_train,X_test):
        mean=X_train.mean(axis=0)
        std=X_train.std(axis=0)+1e-8
        X_train=(X_train-mean)/std
        X_test=(X_test-mean)/std
        return X_train,X_test
    def get_data(self):
       
        X_train,X_test,y_train,y_test=train_test_split(
            self.X,self.y,test_size=self.test_size,
            random_state=self.random_state)
        X_train,X_test=self.normalize(X_train,X_test)
        return X_train,X_test,y_train,y_test
class Sigmoid:
    @staticmethod
    def forward(z):
        z_clipped = np.clip(z, -500, 500)
        return 1/(1+np.exp(-z_clipped))
    @staticmethod
    def backward(z):
        s=Sigmoid.forward(z)
        return s*(1-s)
class BinaryCrossEntropy:
    @staticmethod
    def forward(y_true,y_pred):
        eps=1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true*np.log(y_pred)+ (1-y_true)*np.log(1-y_pred))

class Perceptron:
    def __init__(self,input_features):
        self.w=np.random.randn(input_features,1)*0.01
        self.b=np.zeros((1,1))
        self.X=None
        self.A=None
        self.Z=None
        self.dw=None
        self.db=None
    def forward(self,X):
        self.X=X
        self.Z=np.dot(X,self.w)+self.b
        self.A=Sigmoid.forward(self.Z)
        return self.A
    def backward(self,y_true):
        m=self.X.shape[0]
        dz = self.A - y_true
        self.dw=(1/m)*np.dot(self.X.T,dz)
        self.db=(1/m)*np.sum(dz,axis=0, keepdims=True)
        return dz
    def update(self,lr=0.01):
        self.w -=lr*self.dw
        self.b -=lr*self.db
    def predict(self, X):
        A = self.forward(X)
        return (A > 0.5).astype(int)

class Trainer:
    def __init__(self,Model,loss_fn,lr=0.01):
        self.Model=Model
        self.loss_fn=loss_fn
        self.lr=lr
        self.history_loss=[]
    def Train(self,X,y,epochs=2000):
        for epoch in range(epochs):
            y_pred=self.Model.forward(X)
            loss=self.loss_fn.forward(y,y_pred)
            self.history_loss.append(loss)
           
            self.Model.backward(y)
            self.Model.update(self.lr)

            if epoch%100==0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return self.history_loss

dataset=BreastCancerDataset()
X_train,X_test,y_train,y_test=dataset.get_data()

model=Perceptron(input_features=30)
trainer=Trainer(Model=model,loss_fn=BinaryCrossEntropy,lr=0.1)

history_loss=trainer.Train(X_train,y_train,epochs=2000)
preds = model.predict(X_test)
accuracy = np.mean(preds == y_test)
print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
print("Predictions:")
print(preds)

plt.plot(history_loss)
plt.title("Single Neuron training")
plt.xlabel("Epoches")
plt.ylabel("loss")
plt.show()
  