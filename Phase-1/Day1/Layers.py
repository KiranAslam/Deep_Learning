import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()   #for creating data 

X=[[1,2,3,2.5],
        [2.0,5.0,-1.0,2.0],
        [-1.5,2.7,3.3,-0.8]]

X,y=spiral_data(100,3)


class Dense_Layer:
    def __init__(self, n_inputs,n_neuron):
        self.weights=0.10*np.random.rand(n_inputs,n_neuron)
        self.biases=np.zeros((1,n_neuron))

    def forward(self,inputs):
        self.output=np.dot(inputs,self.weights)+ self.biases

class Activation_ReLU:
    def forward(self,inputs):
        self.output= np.maximum(0,inputs)


Layer1=Dense_Layer(2,5)
activation1=Activation_ReLU()
Layer1.forward(X)
activation1.forward(Layer1.output)
print(activation1.output)

#Layer2=Dense_Layer(5,2)
#Layer2.forward(Layer1.output)
#print(Layer2.output)