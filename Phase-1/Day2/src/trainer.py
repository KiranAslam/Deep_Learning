import numpy as np

class Trainer:
    def __init__(self,X,model,loss_fn,lr=0.01):
        self.model=model
        self.loss_fn=loss_fn
        self.lr=lr
        self.loss_history=[]
    def train(self,X,y,epochs=1000):
        for epoch in range(epochs):

            y_pred=self.model.forward(X)
            loss=self.loss_fn.forward(y,y_pred)
            self.loss_history.append(loss)

            dA=self.loss_fn.backward(y,y_pred)
            self.model.backward(dA)

            self.model.update(self.lr)

            if epoch % 100 ==0:
                print(f"Epoch : {epoch} | loss: {loss:.4f}")
        return self.loss_history


