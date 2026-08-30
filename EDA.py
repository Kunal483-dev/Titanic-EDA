
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

file_path ="Titanic-Cleaned-Dataset.csv"
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nNumerical feature summary:")
print(df.describe())

print("\nMedian values:")
print(df[["Age", "Fare", "SibSp", "Parch"]].median())

print("\nCategorical feature summary:")
print(df[["Sex", "Embarked"]].describe())

numeric_features = ["Age", "Fare", "SibSp", "Parch"]

df[numeric_features].hist(
    bins=20,
    figsize=(12, 8),
    edgecolor="black"
)

plt.suptitle("Distribution of Numeric Features", fontsize=16)
plt.tight_layout()
plt.show()