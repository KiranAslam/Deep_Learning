import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()   #for creating data 


X,y=spiral_data(samples=100,classes=3)


class Dense_Layer:
    def __init__(self, n_inputs,n_neuron):
        self.weights=0.10*np.random.rand(n_inputs,n_neuron)
        self.biases=np.zeros((1,n_neuron))

    def forward(self,inputs):
        self.output=np.dot(inputs,self.weights)+ self.biases

class Activation_ReLU:
    def forward(self,inputs):
        self.output= np.maximum(0,inputs)

class Activation_softmax:
    def forward(self,inputs):
        exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        probabilities=exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output=probabilities
class Loss:
    def calculate(self,output,y):
        sample_losses=self.forward(output,y)
        data_loss=np.mean(sample_losses)
        return data_loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        samples=len(y_pred)
        y_pred_clipped=np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape)==1:
            correct_confidences=y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape)==2:
            correct_confidences=np.sum(y_pred_clipped*y_true,axis=1)
        negative_log_likelihood= -np.log(correct_confidences)
        return negative_log_likelihood

Layer1=Dense_Layer(2,3)
activation1=Activation_ReLU()
Layer1.forward(X)
activation1.forward(Layer1.output)
print("====ReLU activation=====")
print(activation1.output[:5])

Layer2=Dense_Layer(3,3)
Activation2=Activation_softmax()
Layer2.forward(Layer1.output)
Activation2.forward(Layer2.output)
print("====Softmax_activation====")
print(Activation2.output[:5])

loss_function=Loss_CategoricalCrossentropy()
loss=loss_function.calculate(Activation2.output,y)
print(f"Loss:",loss)