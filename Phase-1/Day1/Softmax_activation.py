import numpy as np
import math
layer_outputs=[4.8,1.21,2.385]
#layer_outputs=[4.8,4.79,4.25]
#E=2.7182828
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