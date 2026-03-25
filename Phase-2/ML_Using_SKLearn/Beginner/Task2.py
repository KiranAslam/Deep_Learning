import numpy as np
from sklearn.model_selection import train_test_split

X = np.random.rand(100, 5) 
y = np.random.randint(0, 2, 100) 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)