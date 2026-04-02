import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('hotel_bookings.csv')
print(df.info())
print(df.describe())
print(df.head(15))
print(df.isnull().sum())
print(df.duplicated().sum())
df['Arrival_Date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' +                                     df['arrival_date_month'] + '-' + 
                                  df['arrival_date_day_of_month'].astype(str))