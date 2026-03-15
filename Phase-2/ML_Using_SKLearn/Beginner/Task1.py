from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

print(iris.feature_names)
print(iris.target_names)
print(X.shape)
print(y.shape)
print(X[:5])