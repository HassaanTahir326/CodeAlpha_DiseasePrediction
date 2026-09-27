import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_FOLDER = "../results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

df = pd.read_csv("../heart.csv")

# --- Basic structure checks ---
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn info:")
print(df.info())

print("\nAny missing values?")
print(df.isnull().sum())

print("\nHow many patients have heart disease vs not:")
print(df["target"].value_counts())

# --- Correlation heatmap ---
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, "correlation_heatmap.png"))
print(f"\nCorrelation heatmap saved as {RESULTS_FOLDER}/correlation_heatmap.png")

# --- Feature distributions split by target ---
# Compares patients with vs without disease across a few key continuous features.
# Map target to readable labels first so the legend can never get mismatched.
df_plot = df.copy()
df_plot["Diagnosis"] = df_plot["target"].map({0: "No Disease", 1: "Disease"})

features_to_plot = ["age", "chol", "trestbps", "thalach"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(features_to_plot):
    sns.histplot(
        data=df_plot, x=feature, hue="Diagnosis",
        kde=True, element="step", ax=axes[i],
        palette={"No Disease": "steelblue", "Disease": "indianred"}
    )
    axes[i].set_title(f"{feature} distribution by disease status")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, "feature_distributions.png"))
print(f"Feature distributions saved as {RESULTS_FOLDER}/feature_distributions.png")