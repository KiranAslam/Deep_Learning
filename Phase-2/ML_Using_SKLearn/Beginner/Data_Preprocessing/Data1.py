import pandas as pd
import numpy as np
import random
import json

# --- 1. CSV File Banayein (Transaction Data) ---
n_rows = 1000
data_csv = {
    'Customer_ID': [f'CUST_{i}' for i in range(1, n_rows + 1)],
    'Transaction_Amount': [random.uniform(10, 5000) if i % 15 != 0 else -999 for i in range(n_rows)], # Negative values (outliers)
    'Date': [random.choice(['2023-01-01', '05/12/2023', '2023.06.15', 'Oct 10, 2023']) for _ in range(n_rows)], # Messy dates
    'Product_Category': [random.choice(['Electronics', 'Clothing', 'Home', None]) for _ in range(n_rows)] # Missing values
}
df_csv = pd.DataFrame(data_csv)
df_csv.to_csv('Main_Data.csv', index=False)

# --- 2. JSON File Banayein (Customer Details) ---
# Sirf 800 customers rakhenge taake Join ke waqt 200 missing ho jayein
customer_details = []
for i in range(1, 801):
    customer_details.append({
        'Customer_ID': f'CUST_{i}',
        'Age': f"{random.randint(18, 70)} years", # String format "25 years"
        'City': random.choice(['Karachi', 'Lahore', 'Islamabad', 'Peshawar']),
        'Signup_Date': '2022-01-01'
    })

with open('Customer_Details.json', 'w') as f:
    json.dump(customer_details, f)

print("Both files 'Main_Data.csv' and 'Customer_Details.json' created successfully!")