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