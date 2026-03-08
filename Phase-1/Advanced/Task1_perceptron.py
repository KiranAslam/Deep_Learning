import numpy as np
import matplotlib.pyplot as plt

X=np.array([  [0, 0], 
              [0, 1], 
              [1, 0], 
              [1, 1]])
y=np.array([[0], [0], [0], [1]])
class Perceptron:
    def __init__(self,input_features):
        self.w=np.random.rand(input_features,1)*0.01
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
    def backward(self,dA):
        m=self.X.shape[0]
        dz=dA*Sigmoid.backward(self.Z)
        self.dw=(1/m)*np.dot(self.X.T,dz)
        self.db=(1/m)*np.sum(dz)
        return dz
    def update(self,lr=0.01):
        self.w -=lr*self.dw
        self.b -=lr*self.db
    def predict(self, X):
        A = self.forward(X)
        return (A > 0.5).astype(int)
class Sigmoid:
    def forward(z):
        return 1/(1+np.exp(-z))
    def backward(z):
        s=Sigmoid.forward(z)
        return s*(1-s)
class BinaryCrossEntropy:
    def forward(y_true,y_pred):
        eps=1e-8
        return -np.mean(y_true*np.log(y_pred+ eps)+ (1-y_true)*np.log(1-y_pred+eps))
    def backward(y_true,y_pred):
        eps=1e-8
        return (y_pred-y_true)/((y_pred+eps)*(1-y_pred+eps))
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
            dA=self.loss_fn.backward(y,y_pred)
            self.Model.backward(dA)
            self.Model.update(self.lr)

            if epoch%100==0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return self.history_loss
y=y.reshape(-1,1)
model=Perceptron(input_features=2)
trainer=Trainer(Model=model,loss_fn=BinaryCrossEntropy,lr=0.1)
history_loss=trainer.Train(X,y,epochs=2000)
print("Predictions:")
print(model.predict(X))
plt.plot(history_loss)
plt.title("Single Neuron training")
plt.xlabel("Epoches")
plt.ylabel("loss")
plt.show()