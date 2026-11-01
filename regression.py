"""
Tehran House Price Prediction

This project analyzes the relationship between housing features
and property prices in Tehran using different regression models.

It compares different feature encoding techniques, including
One-Hot Encoding and Score-Based Encoding, and evaluates their
impact on prediction performance.

The project also explores logarithmic regression.
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"data\price.csv")
pd.set_option("display.float_format", lambda x: f"{x:.2f}")

print(df.head(10))
print(df.describe())

df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
df["Price(USD)"] = pd.to_numeric(df["Price(USD)"], errors="coerce")

df = df.dropna(subset=["Area", "Room", "Elevator", "Warehouse", "Parking", "Address", "Price(USD)"])

encoded = pd.get_dummies(df["Address"], dtype=int)

plt.scatter(df["Area"], df["Price(USD)"], color="red", label="Area-Price Relationship", alpha=0.4)
plt.xlabel("Area")
plt.ylabel("Price(USD)")
plt.legend()
plt.grid(
    visible=False
)
plt.tight_layout()
plt.show()

top_address = df["Address"].value_counts().head(10).index
plot_data = df[df["Address"].isin(top_address)]

for address, group in plot_data.groupby("Address"):
    plt.scatter(
        group["Area"],
        group["Price(USD)"],
        label=address,
        alpha=0.6
    )

plt.legend()
plt.xlabel("Area")
plt.ylabel("Price(USD)")
plt.grid(alpha=0.6)
plt.tight_layout()
plt.show()
