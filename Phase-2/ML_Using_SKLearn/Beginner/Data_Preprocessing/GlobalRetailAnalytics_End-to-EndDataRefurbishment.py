import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

df_csv = pd.read_csv('Data3.csv')

#Data Auditing
#print('Dataset Info', df_csv.describe())
#print(df_csv.info)
#print(df_csv.head(15))
#print('Null values', df_csv.isnull().sum())
#print("Duplicate values", df_csv.duplicated().sum())
#print(df_csv["Order Date"].head())

df_csv['Order Date'] = pd.to_datetime(df_csv['Order Date'] , dayfirst=True)
df_csv['Month'] = df_csv['Order Date'].dt.month
df_csv['Postal Code'] = df_csv['Postal Code'].fillna(df_csv.groupby('City')['Postal Code'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan))
print(df_csv['Postal Code'].isnull().sum())
df_csv['Product Name'] = df_csv['Product Name'].str.strip().str.lower().str.replace(r'[^\w\s]', '',regex=True).str.title()
df_csv['City'] = df_csv['City'].str.strip().str.lower().str.replace(r'[^\w\s]', '',regex=True).str.title()
df_csv['State'] = df_csv['State'].str.strip().str.lower().str.replace(r'[^\w\s]', '',regex=True).str.title()
df_csv['Category'] = df_csv['Category'].str.strip().str.lower().str.replace(r'[^\w\s]', '',regex=True).str.title()
df_csv['Sub-Category'] = df_csv['Sub-Category'].str.strip().str.lower().str.replace(r'[^\w\s]', '',regex=True).str.title()
df_csv.drop(columns=['Row ID'], inplace=True)

# outlier treatment using IQR method
Q1 = df_csv['Sales'].quantile(0.25)
Q3 = df_csv['Sales'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
df_csv = df_csv[(df_csv['Sales'] >= lower_limit) & (df_csv['Sales'] <=upper_limit)]


plt.figure(figsize=(10,6))
sns.lineplot(data=df_csv,x='Month',y='Sales',estimator='sum',errorbar=None)
plt.title('Sales trend over month ')
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(df_csv['Sales'], bins=50,kde=True,color='teal')
plt.title('Sales Distributions')
plt.show()

plt.figure(figsize=(12,6))
sns.countplot(data=df_csv, x='Category' , hue='Sub-Category', palette='Set2')
plt.title('Most sold categories')
plt.show()

print(df_csv.head(10))
print('Null values', df_csv.isnull().sum())
print("Duplicate values", df_csv.duplicated().sum())
