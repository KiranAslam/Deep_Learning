from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

print(cm)