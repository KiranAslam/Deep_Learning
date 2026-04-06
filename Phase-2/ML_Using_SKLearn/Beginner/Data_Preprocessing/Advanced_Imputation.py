import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer


df = pd.read_csv('diabetes.csv')

#print(df.describe())
#print(df.info())
#print(df.head(10))
#print(f"missing values: { df.isnull().sum()}")
#print(f"Duplicate values: { df.duplicated().sum()}")

cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0,np.nan)
print(f"missing values: { df.isnull().sum()}")
print(df.describe())
print(df.info())
print(df.head(10))