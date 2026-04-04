import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

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
df['Total_stays'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
month_map = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 
             'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}
df['arrival_year_date_month_num'] = df['arrival_date_month'].map(month_map)
df['arrival_date'] = pd.to_datetime(df[['arrival_date_year', 'arrival_year_date_month_num', 'arrival_date_day_of_month']].astype(str).agg('-'.join, axis=1))
print(df[['Total_stays', 'arrival_date', 'Total_guests']].head())



plt.figure(figsize=(10,6))
sns.countplot(data=df, x='hotel', hue='is_canceled', palette='magma')
plt.title('Distribution of Canceled vs Not Canceled Bookings')
plt.xlabel("Hotel Type")
plt.ylabel("Number of bookings")
plt.legend(title='Is Canceled', labels=['No', 'Yes'])
#plt.show()

sns.barplot(data=df, x='hotel', hue='booking_changes', estimator='mean', palette='viridis')
plt.title('Average Booking Changes by Hotel')
plt.xlabel("Hotel")
plt.ylabel("Average Booking Changes")
#plt.show()

hotel_counts = df['hotel'].value_counts()
plt.figure(figsize=(7,7))
plt.pie(hotel_counts, labels=hotel_counts.index, autopct='%1.1f%%', startangle=140, colors=['teal', 'orange'])
plt.title('Market Share of Hotels')
#plt.show()

plt.figure(figsize=(12,6))
month_order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
df['arrival_date_month'] = pd.Categorical(df['arrival_date_month'], categories=month_order, ordered=True)
sns.lineplot(data=df, x='arrival_date_month', y='adr', marker='o')
plt.xticks(rotation=45)
plt.title("Average Daily Rate by Arrival Month")
plt.xlabel("Arrival Month")
plt.ylabel("Average Daily Rate")
#plt.show()

sns.boxplot(data=df, x='hotel', y='adr', palette='Set2')
plt.title("Boxplot of Average Daily Rate by Hotel")
#plt.show()

df.drop(columns=['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'], inplace=True)
df=pd.get_dummies(df,columns=['hotel', 'meal', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type'], drop_first=True)
X= df.drop(columns=['is_canceled','arrival_date', 'country','reservation_status'])
y=df['is_canceled']
X = X.astype(float)
print(df.info())
print(df.head(15))
print(f"Missing values: {df.isnull().sum()}")
print(f"duplicated values: {df.duplicated().sum()}")

x_train, x_test,y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
print(f"Total data: {X.shape[0]}")
print(f"Training data: {x_train.shape[0]}")
print(f"Testing data: {x_test.shape[0]}")

scalar = StandardScaler()
x_train_scaled = scalar.fit_transform(x_train)
x_test_scaled = scalar.transform(x_test)

Lr_model = LogisticRegression()
lr_model = lr_model.fit(x_train_scaled, y_train)
lr_predictions = lr_model.predict(x_test_Scaled)

print("Logistic Regression Classification Report:")
print(f"Accuracy: {accuracy_score(y_test, lr_predictions)* 100 : .2f}%")
print(classification_report(y_test, lr_predictions))

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model = rf_model.fit(x_train_scaled, y_train)
rf_predictions = rf_model.predict(x_test_scaled)

print("Random Forest Classification Report:")
print(f"Accuracy: {accuracy_score(y_test, rf_predictions)* 100 : .2f}%")
print(classification_report(y_test, rf_predictions))