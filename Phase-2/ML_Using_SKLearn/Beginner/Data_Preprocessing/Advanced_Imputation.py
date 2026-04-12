import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('diabetes.csv')
#print(df.describe())
#print(df.info())
#print(df.head(10))
#print(f"missing values: { df.isnull().sum()}")
#print(f"Duplicate values: { df.duplicated().sum()}")
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0,np.nan)
print(f"missing values: { df.isnull().sum()}")
#print(df.describe())
#print(df.info())
#print(df.head(10))
X = df.drop('Outcome', axis=1)
Y = df['Outcome']
# Simpler Imputer
simpler_imputer = SimpleImputer(strategy='median')
X_simpler = simpler_imputer.fit_transform(X)
#Knn Imputer
knn_imputer = KNNImputer(n_neighbors=5)
X_knn = knn_imputer.fit_transform(X)
# Iterative Imputer
iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
X_tier = iterative_imputer.fit_transform(X)

df_simple = pd.DataFrame(X_simpler,columns=X.columns)
df_knn = pd.DataFrame(X_knn,columns=X.columns)
df_tier = pd.DataFrame(X_tier,columns=X.columns)

print("Insuline camparison")
compare_insulin = pd.DataFrame({
    'original' : X['Insulin'].head(10).values,
    'Simpler ' : df_simple['Insulin'].head(10).values,
    'Knn_imputer' : df_knn['Insulin'].head(10).values,
    'Interative' : df_tier['Insulin'].head(10).values
})
print(compare_insulin)

datasets={
    'Simpler': df_simple,
    'Knn_imputer': df_knn,
    'Iterative': df_tier
}

print("Accuracy comparison:")

for name,data in datasets.items():
    scalar=StandardScaler()
    x_scaled=scalar.fit_transform(data)
    model=LogisticRegression()
    score=cross_val_score(model,x_scaled,Y,cv=5).mean()
    print(f"{name} : {score.mean():.4f}")