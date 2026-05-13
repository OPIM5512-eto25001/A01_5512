from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Make a boxplot
plt.figure(figsize=(8, 6))
df.boxplot(column=["MedInc"])

# Add title
plt.title("Boxplot of Median Income")

# Save figure
plt.savefig("figs/boxplot.png")

# Show plot
plt.show()