import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = sns.load_dataset("titanic")

print(df.head())
print(df.info())
print(df.isnull().sum())

df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

scaler = StandardScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])


sns.boxplot(data=df[['age', 'fare']])
plt.show()


df.to_csv("cleaned_titanic.csv", index=False)
print("Data cleaned and saved successfully!")
cd path/to/your/project
git init




