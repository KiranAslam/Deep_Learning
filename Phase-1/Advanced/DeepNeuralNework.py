import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split

class MNIST_Dataset:
    def __init__(self,random_state=42):
        self.random_state=random_state
        data = sklearn.datasets.fetch_openml(
            'mnist_784',
            version=1,
            as_frame=False,
            parser='liac-arff'
           )
        self.X=data.data
        self.y=data.target.astype(np.float32)
 
    def normalize(self, X_train, X_test, X_val):
            mean = X_train.mean(axis=0)
            std = X_train.std(axis=0) + 1e-8
            X_train = (X_train - mean) / std

            X_test = (X_test - mean) / std 
            X_val = (X_val - mean) / std
            return X_train, X_test, X_val

    def one_hot_encoding(self,y):
        num_class=10
        one_hot=np.zeros((len(y),num_class))
        one_hot[np.arange(len(y)),y.astype(int)]=1
        return one_hot
    def train_data(self):
        return self.X_train,self.y_train
    def vald_data(self):
        return self.X_val,self.y_val
    def test_data(self):
        return self.X_test,self.y_test
    def get_data(self):
        #split : train+test
        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(
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
        self.X_train,self.X_test,self.X_val=self.normalize(
            self.X_train,self.X_test,self.X_val
        )
        #one hot encoding
        self.y_train=self.one_hot_encoding(self.y_train)
        self.y_test=self.one_hot_encoding(self.y_test)
        self.y_val=self.one_hot_encoding(self.y_val)

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

        self.w1=np.random.randn(input_size,hidden_size1)*np.sqrt(2.0 / input_size)
        self.b1=np.zeros((1,hidden_size1))

        self.w2=np.random.randn(hidden_size1,hidden_size2)*np.sqrt(2.0 / input_size)
        self.b2=np.zeros((1,hidden_size2))

        self.w3=np.random.randn(hidden_size2,output_size)*np.sqrt(2.0 / input_size)
        self.b3=np.zeros((1,output_size))

    def forward(self,X):
        self.X=X
        self.Z1=np.dot(X,self.w1)+self.b1
        self.A1=ReLU.forward(self.Z1)

        self.Z2=np.dot(self.A1,self.w2)+self.b2
        self.A2=ReLU.forward(self.Z2)

        self.Z3=np.dot(self.A2,self.w3)+self.b3
        self.A3=Softmax.forward(self.Z3)

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
        dw1=(1/m)*np.dot(self.X.T,dZ1)
        db1=(1/m)*np.sum(dZ1,axis=0,keepdims=True)
        self.dw3 = dw3
        self.db3 = db3
        self.dw2 = dw2
        self.db2 = db2
        self.dw1 = dw1
        self.db1 = db1
    def update(self,lr=0.01):
        self.w3 -= lr*self.dw3
        self.b3 -= lr*self.db3
        self.w2 -= lr*self.dw2
        self.b2 -= lr*self.db2
        self.w1 -= lr*self.dw1
        self.b1 -= lr*self.db1

    def predict(self,X):
        probs=self.forward(X)
        return np.argmax(probs,axis=1)

class Trainer:
    def __init__(self,model,dataset,lr=0.01,epochs=20,batch_size=64):

        self.model = model
        self.dataset = dataset
        
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.train_losses = []
        self.val_losses = []
        self.train_acc = []
        self.val_acc = []

    def get_batches(self,X,y):
        n=X.shape[0]
        indices=np.arange(n)
        np.random.shuffle(indices)

        for start in range(0,n,self.batch_size):
            end=start+self.batch_size 
            batch_indices=indices[start:end]
            yield X[batch_indices], y[batch_indices]    
    def Accuracy(self,y_true,y_pred):
        y_true=np.argmax(y_true,axis=1)
        return np.mean(y_true==y_pred)
    def train(self):
        X_train,y_train=self.dataset.train_data()
        X_val,y_val=self.dataset.vald_data()

        for epoch in range(self.epochs):
            epoch_loss=0
            batch_count=0

            for X_batch,y_batch in self.get_batches(X_train,y_train):
                probs=self.model.forward(X_batch)
                loss=CrossEntropyLoss.forward(y_batch,probs)
                epoch_loss += loss

                self.model.backward(y_batch)
                self.model.update(self.lr)

                batch_count +=1

            avg_loss= epoch_loss/batch_count
            self.train_losses.append(avg_loss)

            val_probs=self.model.forward(X_val)
            val_loss=CrossEntropyLoss.forward(y_val,val_probs)
            self.val_losses.append(val_loss)

            train_preds=self.model.predict(X_train)
            val_preds=self.model.predict(X_val)
            train_acc=self.Accuracy(y_train,train_preds)
            val_acc=self.Accuracy(y_val,val_preds)

            self.train_acc.append(train_acc)
            self.val_acc.append(val_acc)

            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

    def plot_loss(self):

        plt.figure(figsize=(6,4))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.show()

    def plot_accuracy(self):

        plt.figure(figsize=(6,4))
        plt.plot(self.train_acc, label="Train Accuracy")
        plt.plot(self.val_acc, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()
        plt.show()


dataset=MNIST_Dataset()
dataset.get_data()
model=DeepNeuralNetwork()
trainer = Trainer(
    model=model,
    dataset=dataset,
    lr=0.01,
    epochs=20,
    batch_size=64
)

trainer.train()
trainer.plot_loss()
trainer.plot_accuracy()
X_test, y_test = dataset.test_data()
preds = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
test_accuracy = np.mean(preds == y_true)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")