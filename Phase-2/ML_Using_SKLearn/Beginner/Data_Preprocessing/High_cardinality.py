import numpy as np
import pandas as pd 

df = pd.read_csv('hotel_bookings.csv')

country_means = df.groupby('country')['is_canceled'].mean()
df['country_encoded'] = df['country'].map(country_means)
  
print(df[['country', 'is_canceled', 'country_encoded']].head(10))