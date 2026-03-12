import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm


digits= datasets.load_digits()
#print(digits.data)
#print(digits.target)
print(digits.images)
print(len(digits.data))
clf=svm.SVC(gamma=0.001,C=100)
x,y= digits.data[:-10], digits.target[:-10]
clf.fit(x,y)
pred = clf.predict(digits.data[-2:-1])
print(f"prediction: {pred}")

plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()