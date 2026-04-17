import numpy as np
import pandas as pd 
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


df = pd.read_csv('hotel_bookings.csv')
df.drop(columns=['company','agent', 'reservation_status_date'], inplace=True)
df['children'] = df['children'].fillna(0)
df['country'] = df['country'].fillna('Unknown')


Q1, Q3 = df['adr'].quantile(0.25), df['adr'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['adr'] >= Q1 - 1.5*IQR) & (df['adr'] <= Q3 + 1.5*IQR)]
df.drop_duplicates(inplace=True)


df['Total_guests'] = df['adults'] + df['children'] + df['babies']
df['Total_stays'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
month_map = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 
             'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}
df['arrival_year_date_month_num'] = df['arrival_date_month'].map(month_map)


X = df.drop(columns=['is_canceled', 'reservation_status']) 
y = df['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


encoder = ce.TargetEncoder(cols=['country'], smoothing=10)
X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)

X_train = X_train.drop(columns=['arrival_date_month']) 


numeric_cols = X_train.select_dtypes(include=[np.number]).columns
X_train_final = X_train[numeric_cols]
X_test_final = X_test[numeric_cols] 
smote = SMOTE(random_state=42)
X_train_res,y_train_res = smote.fit_resample(X_train_final, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_res, y_train_res)

y_pred = rf_model.predict(X_test_final)

print(f"\n Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')