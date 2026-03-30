import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_csv = pd.read_csv('Scraped_Data.csv')
#df_csv.info()
#print(df_csv.head(10))
df_csv['Product_Name'] = df_csv['Product_Name'].str.strip().str.replace(r'[^\w\s]', '', regex=True).str.title()
df_csv.drop_duplicates(inplace=True)
print(df_csv.duplicated().sum())
df_csv = df_csv[df_csv['Price'] < 5000]
df_csv['Rating'] = df_csv['Rating'].fillna(df_csv.groupby('Product_Name')['Rating'].transform('mean'))
print('Null values in Rating column:', df_csv['Rating'].isna().sum())
print(df_csv.head(20))
plt.figure(figsize=(10,6))
sns.boxplot(x=df_csv['Price'])
plt.title('Boxplot of Product Prices')
plt.show()

corr= df_csv[['Price','Rating','Stock']].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr,annot=True, cmap='RdYlGn')
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df_csv['Price'], bins=30, kde=True , color='purple')
plt.title('Distribution of Product Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

sns.pairplot(df_csv, hue='Product_Name', palette='husl')
plt.show()
