import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('hotel_bookings.csv')
#print(df.info())
#print(df.describe())
#print(df.head(15))
#print(df.isnull().sum())
#print(df.duplicated().sum())


df.drop(columns=['company','agent', 'reservation_status_date'], inplace=True)
df['children']=df['children'].fillna(0)
df['country']=df['country'].fillna('Unknown')
Q1= df['adr'].quantile(0.25)
Q3= df['adr'].quantile(0.75)
IQR = Q3-Q1
lower_bound = Q1 -1.5*IQR
upper_bound =Q3 + 1.5*IQR
df = df[(df['adr'] >= lower_bound) & (df['adr'] <= upper_bound)]
df.drop_duplicates(inplace=True)
df['Total_guests'] = df['adults'] + df['children'] + df['babies']
df['total_stays'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
month_map = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 
             'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}
df['arrival_year_date_month_num'] = df['arrival_date_month'].map(month_map)
df['arrival_date'] = pd.to_datetime(df[['arrival_date_year', 'arrival_year_date_month_num', 'arrival_date_day_of_month']].astype(str).agg('-'.join, axis=1))
print(df[['total_stays', 'arrival_date', 'total_guests']].head())
#print(df.info())
#print(df.describe())
#print(df.head(15))
#print(df.isnull().sum())
#print(df.duplicated().sum())

plt.figure(figsize=(10,6))
sns.countplot(data=df, x='hotel', hue='is_canceled', palette='magma')
plt.title('Distribution of Canceled vs Not Canceled Bookings')
plt.xlabel("Hotel Type")
plt.ylabel("Number of bookings")
plt.legend(title='Is Canceled', labels=['No', 'Yes'])
plt.show()

sns.barplot(data=df, x='hotel', hue='booking_changes', estimator='mean', palette='viridis')
plt.title('Average Booking Changes by Hotel')
plt.xlabel("Hotel")
plt.ylabel("Average Booking Changes")
plt.show()

hotel_counts = df['hotel'].value_counts()
plt.figure(figsize=(7,7))
plt.pie(hotel_counts, labels=hotel_counts.index, autopct='%1.1f%%', startangle=140, colors=['teal', 'orange'])
plt.title('Market Share of Hotels')
plt.show()

plt.figure(figsize=(12,6))
month_order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
df['arrival_date_month'] = pd.Categorical(df['arrival_date_month'], categories=month_order, ordered=True)
sns.lineplot(data=df, x='arrival_date_month', y='adr', marker='o')
plt.xticks(rotation=45)
plt.title("Average Daily Rate by Arrival Month")
plt.xlabel("Arrival Month")
plt.ylabel("Average Daily Rate")
plt.show()

sns.boxplot(data=df, x='hotel', hue='adr', palette='Set2')
plt.title("Boxplot of Average Daily Rate by Hotel")
plt.show()

