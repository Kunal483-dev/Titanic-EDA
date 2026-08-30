
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

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(y=df["Age"], ax=axes[0], color="skyblue")
axes[0].set_title("Boxplot of Age")

sns.boxplot(y=df["Fare"], ax=axes[1], color="salmon")
axes[1].set_title("Boxplot of Fare")

plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Sex", hue="Survived", palette="Set2")
plt.title("Survival Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

print(df.groupby("Sex")["Survived"].mean())

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Pclass", hue="Survived", palette="Set1")
plt.title("Survival Count by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

print(df.groupby("Pclass")["Survived"].mean())

plt.figure(figsize=(7, 5))

sns.barplot(data=df, x="Embarked", y="Survived", errorbar=None, palette="viridis")
plt.title("Average Survival Rate by Embarkation Port")
plt.xlabel("Embarkation Port")
plt.ylabel("Survival Rate")
plt.show()