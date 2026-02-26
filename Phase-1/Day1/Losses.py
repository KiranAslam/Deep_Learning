import numpy as np
import math
# catagorical cross entropy loss

softmax_output=[0.7,0.7,0.4]
targeted_output=[1,0,0]# one hot encoding

loss=-math.log(softmax_output[0]*targeted_output[0])
print(loss)