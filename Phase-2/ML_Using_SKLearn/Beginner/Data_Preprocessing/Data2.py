import random
import pandas as pd
import numpy as np

n = 600
new_data = {
    'Product_Name': [random.choice([' iPhone ', 'Laptop!!!', 'TV...', '  Watch  ', 'Charger', 'Mous-e']) for _ in range(n)],
    'Price': [random.uniform(100, 1500) if i % 30 != 0 else 75000 for i in range(n)], 
    'Rating': [random.uniform(1, 5) if i % 12 != 0 else np.nan for i in range(n)], 
    'Stock': [random.randint(0, 500) for _ in range(n)]
}
df_scrap = pd.DataFrame(new_data)
df_scrap = pd.concat([df_scrap, df_scrap.head(20)], ignore_index=True)

df_scrap.to_csv('Scraped_Data.csv', index=False)

print("Dataset is ready!")