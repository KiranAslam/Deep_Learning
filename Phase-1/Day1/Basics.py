'''
#Single neuron with 3 inputs and one output
input=[1,2,3]
weights=[0.2,0.3,0.4]
bias=2
Output1=input[0]*weights[0]+input[1]*weights[1]+input[2]*weights[2]+bias
print(Output1)
'''

'''

# 4 inputs and two output
inputs=[1,2,3,4]
weight1=[0.2,0.3,0.4,0.6]
weight2=[0.1,0.5,-0.2,0.5]

bias=2
bias=3

Output2=[inputs[0]*weight1[0]+inputs[1]*weight1[1]+inputs[2]*weight1[2]+inputs[3]*weight1[3]+bias,
        inputs[0]*weight2[0]+inputs[1]*weight2[1]+inputs[2]*weight2[2]+inputs[3]*weight1[3]+bias]

print(Output2)

'''

'''
#more Better way 
inputs=[1,2,3,4]
all_weights=[[0.2,0.3,0.4,0.6],[0.1,0.5,-0.2,0.5],[0.5,0.1,0.25,0.5]]
biases=[2,3,0.5]

layer_output=[]#output of current layer
for neuron_weights, neuron_bias in zip(all_weights, biases):
    output=0#output of given neuron
    for input_values,weights in zip(inputs,neuron_weights):
        output += input_values*weights
    output += neuron_bias
    layer_output.append(output)

print(layer_output)

'''
'''
# simpler way using numpy
import numpy as np

inputs=[1,2,3,4]
weights=[0.2,0.5,0.1,0.4]
bias=2
output=np.dot(weights,inputs)+bias
print(output)
'''
import numpy as np 
# dot product of layers of neuron
inputs=[1,2,3,4]
weights=[[0.2,0.3,0.4,0.6],
         [0.1,0.5,-0.2,0.5],
         [0.5,0.1,0.25,0.5]]
biases=[2,3,0.5]
output=np.dot(weights,inputs)+biases
print(output)


