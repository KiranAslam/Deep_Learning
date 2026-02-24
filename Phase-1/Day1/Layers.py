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