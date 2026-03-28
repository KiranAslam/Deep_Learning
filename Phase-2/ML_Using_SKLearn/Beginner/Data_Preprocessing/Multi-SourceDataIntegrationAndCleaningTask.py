import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_csv= pd.read_csv('Main_Data.csv')
df_json= pd.read_json("Customer_Details.json")

#print(df_csv.head())
#print(df_json.head())
#df_csv.info()
#df_json.info()

df_json['Age'] = df_json['Age'].str.replace(' years', '').astype(int)
Most_frequent_Date= df_csv['Date'].mode()[0]
df_csv['Date'] = pd.to_datetime(df_csv['Date'], dayfirst=True, errors='coerce').fillna(Most_frequent_Date)
df_json['Signup_Date'] = pd.to_datetime(df_json['Signup_Date'], dayfirst=True, errors='coerce')
median_value = df_csv[df_csv['Transaction_Amount'] > 0]['Transaction_Amount'].median()
df_csv.loc[df_csv['Transaction_Amount']<0 ,'Transaction_Amount'] = median_value

#df_csv['Product_Category'] = df_csv['Product_Category'].fillna('unknown') 

print(df_csv.head(10))
#print(df_json.head(10))
Customer_Details = df_json.merge(df_csv, on='Customer_ID', how='left')
print(Customer_Details.head(10))
print(Customer_Details.info())

v_counts= Customer_Details['Product_Category'].value_counts(normalize=True)
categories=v_counts.index
probabilities=v_counts.values
missing_mask = Customer_Details['Product_Category'].isna()
Customer_Details.loc[missing_mask,'Product_Category']=np.random.choice(
    categories,size=missing_mask.sum(),p=probabilities
) 

plt.style.use('ggplot')
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
city_spending = Customer_Details.groupby('City')['Transaction_Amount'].sum().sort_values()
city_spending.plot(kind='barh', color='teal')
plt.title('Total Spending by City')
plt.xlabel('Total Spending')


plt.subplot(2,2,2)
sns.scatterplot(data=Customer_Details,x='Age',y='Transaction_Amount',hue='City', alpha=0.6)
plt.title('Age vs Transaction Amount')

plt.subplot(2,2,3)
sns.countplot(data=Customer_Details,x='Product_Category', palette='viridis')
plt.title('Popularity of Product Categories')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()