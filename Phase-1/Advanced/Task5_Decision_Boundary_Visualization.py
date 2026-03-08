import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split

class Iris_dataset:
    def __init__(self,test_size=0.20,random_state=42):
        self.test_size=test_size
        self.random_state=random_state
        data=sklearn.datasets.load_iris()
        self.X=data.data[:, :2]
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
    def forward(y_true,y_pred):
        eps=1e-15
        y_pred=np.clip(y_pred,eps,1-eps)
        return -np.mean(np.sum(y_true*np.log(y_pred),axis=1))

class Neural_Network:
    def __init__(self,input_size=2,hidden_size=8,output_size=3):

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

        self.train_loss=[]
        self.test_loss=[]
        self.train_acc=[]
        self.test_acc=[]

    def Accuracy(self,preds,y_true):
        y_true=np.argmax(y_true,axis=1)
        return np.mean(preds==y_true)
    def Train(self,X_Train,y_Train,X_test,y_test,epochs=4000):
        for epoch in range(epochs):
            #Forward
            y_pred=self.model.forward(X_Train)
            loss=self.loss_fn.forward(y_Train,y_pred)
            #Backward
            self.model.backward(y_Train)
            self.model.update(self.lr)
            #Train accuracy
            train_preds=self.model.predict(X_Train)
            train_accuracy=self.Accuracy(train_preds,y_Train)
            #Test evaluation
            test_preds_probs=self.model.forward(X_test)
            test_loss=self.loss_fn.forward(y_test,test_preds_probs)

            test_preds=np.argmax(test_preds_probs,axis=1)
            test_accuracy=self.Accuracy(test_preds,y_test)

            self.train_loss.append(loss)
            self.test_loss.append(test_loss)
            self.train_acc.append(train_accuracy)
            self.test_acc.append(test_accuracy)

            if epoch % 200==0:
                print(
                    f"Epoch {epoch} | "
                    f"Train Loss: {loss:.4f} | "
                    f"Test Loss: {test_loss:.4f} | "
                    f"Train Acc: {train_accuracy*100:.2f}% | "
                    f"Test Acc: {test_accuracy*100:.2f}%"   
                )
        return self.train_loss,self.test_loss

class DecisionBoundaryVisualizer:
    def __init__(self,model):
        self.model=model
  
    def plot(self,X,y):
        x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
        y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
        xx,yy=np.meshgrid(np.arange(x_min,x_max,0.01),np.arange(y_min,y_max,0.01))
        grid_points=np.c_[xx.ravel(),yy.ravel()]
        Z=self.model.predict(grid_points)
        Z=Z.reshape(xx.shape)
        plt.contourf(xx,yy,Z,cmap=plt.cm.Spectral,alpha=0.8)
        plt.scatter(X[:,0],X[:,1],c=np.argmax(y,axis=1),edgecolors='k',marker='o',s=100,cmap=plt.cm.Spectral)
        plt.title("Decision Boundary")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()



dataset=Iris_dataset()
X_train, X_test, y_train, y_test=dataset.get_data()
model=Neural_Network()
trainer=Trainer(model=model,loss_fn=CategoricalCrossEntropy,lr=0.01)
train_loss,test_loss=trainer.Train(
    X_train,y_train,
    X_test,y_test,
    epochs=4000
)


preds=model.predict(X_test)
y_true=np.argmax(y_test,axis=1)
Accuracy=np.mean(preds==y_true)
print(f"Accuracy: {Accuracy*100:.2f}%")

plt.plot(trainer.train_loss, label="Train loss")
plt.plot(trainer.test_loss, label="Test Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(trainer.train_acc,label="Training Accuracy")
plt.plot(trainer.test_acc,label="Test Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

visualizer = DecisionBoundaryVisualizer(model)
visualizer.plot(X_train, y_train)