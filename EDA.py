
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

file_path = r"C:\Users\rajpo\Desktop\Titanic-Cleaned-Dataset.csv"
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())