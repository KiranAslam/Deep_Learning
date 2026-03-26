import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Task3_HiddenLayer import Neural_Network, ReLU, Softmax, BinaryCrossEntropy, Trainer
from sklearn.metrics import confusion_matrix, classification_report

file_path="fraudTrain.csv"
train_data = pd.read_csv(file_path)
#train_data.info()

fraud=train_data[train_data['is_fraud']==1]
non_fraud=train_data[train_data['is_fraud']==0]

#print(f"total fraud cases: {len(fraud)}")
#print(f"total non-fraud cases: {len(non_fraud)}")

non_fraud_sample= non_fraud.sample(n=20000, random_state=42)
balanced_data = pd.concat([fraud, non_fraud_sample], ignore_index=True)
balanced_data = balanced_data.sample(frac=1,random_state=42).reset_index(drop=True)

features = ['category', 'amt', 'gender', 'city_pop', 'unix_time', 'is_fraud']
balanced_data = balanced_data[features]

balanced_data['gender'] = balanced_data['gender'].map({'M': 0, 'F':1})
balanced_data = pd.get_dummies(balanced_data, columns=['category'], drop_first=True)

#print(f"New shape: {balanced_data.shape}")
#print(balanced_data['is_fraud'].value_counts())
#print(balanced_data.head())
#print(balanced_data.info())

X= balanced_data.drop(('is_fraud'), axis=1)
Y= balanced_data['is_fraud']

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

scalar = StandardScaler()
x_train_scaled= scalar.fit_transform(x_train)
x_test_scaled = scalar.transform(x_test)
print("Data preprocessing completed.")
#print(x_train_scaled[0])


model = Neural_Network(input_size=x_train_scaled.shape[1], hidden_size=16, output_size=2)
def one_hot_fraud(y):
    y = y.astype(int)
    return np.eye(2)[y.flatten()]

y_train_encoded = one_hot_fraud(y_train.values)
y_test_encoded = one_hot_fraud(y_test.values)

trainer = Trainer(model=model, loss_fn=BinaryCrossEntropy, lr=0.05) 
loss_history = trainer.Train(x_train_scaled, y_train_encoded, epochs=1000)

# 5. Accuracy check
preds = model.predict(x_test_scaled)
y_true_labels = np.argmax(y_test_encoded, axis=1)
accuracy = np.mean(preds == y_true_labels)

print(f"Fraud Detection Accuracy: {accuracy*100:.2f}%")
cm = confusion_matrix(y_true_labels, preds)
print("\nConfusion Matrix:")
print(cm)
print("\nDetailed Report:")
print(classification_report(y_true_labels, preds))
