import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

def create_data(points,classes):
    X=np.zeros((points*classes,2)) #data matrix (each row is a single example)
    y=np.zeros(points*classes,dtype='uint8') #class labels
    for class_number in range(classes):
        ix=range(points*class_number,points*(class_number+1))
        r=np.linspace(0.0,1,points) #radius
        t=np.linspace(class_number*4,(class_number+1)*4,points)+np.random.randn(points)*0.2 #theta
        X[ix]=np.c_[r*np.sin(t),r*np.cos(t)]
        y[ix]=class_number
    return X,y

