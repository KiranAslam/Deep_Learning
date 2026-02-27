import numpy as np
import math
import nnfs
nnfs.init()

layer_outputs=[4.8,1.21,2.385]
#layer_outputs=[4.8,4.79,4.25]
#E=2.7182828
#Raw python way 
E=math.e
exp_value=[]
for output in layer_outputs:
    exp_value.append(E**output)

print(exp_value)
#after expontiate the values we need to normalize the values
norm_base=sum(exp_value)
norm_values=[]
for value in exp_value:
    norm_values.append(value / norm_base)
print(norm_values)
print(sum(norm_values))

#using numpy
exp_values=np.exp(layer_outputs)
norm_value=exp_values / np.sum(exp_values)
print(exp_values)
print(sum(norm_value))

#exponentiation+normalization=softmax activation

Layer_outputs=[[4.8,1.21,2.385],
               [8.5,1.2,0.25],
               [1.2,0.25,8.5]]

exp_values=np.exp(Layer_outputs)
norm_values=exp_values/ np.sum(exp_values,axis=1,keepdims=True)
print(np.sum(Layer_outputs,axis=1,keepdims=True))    #axis 1 for rows sum axis 0 fo colunms  keeptims for same shape 
print(exp_values)
print(norm_values)