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
import numpy as np
import sklearn

df = pd.read_csv(r"data\price.csv")
pd.set_option("display.float_format", lambda x: f"{x:.2f}")

print(df.head(10))
print(df.describe())

df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
df["Price(USD)"] = pd.to_numeric(df["Price(USD)"], errors="coerce")

df = df.dropna(subset=["Area", "Room", "Elevator", "Warehouse", "Parking", "Address", "Price(USD)"])

encoded = pd.get_dummies(df["Address"], dtype=int)
df = pd.concat([df, encoded], axis=1)

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

onehot_2f_df = df[["Area", "Price(USD)", *encoded.columns]].copy()

mask = np.random.rand(len(df)) < 0.8
train = onehot_2f_df[mask]
test = onehot_2f_df[~mask]

x_train = np.asanyarray(train[["Area", *encoded.columns]])
y_train = np.asanyarray(train["Price(USD)"])

x_test = np.asanyarray(test[["Area", *encoded.columns]])
y_test = np.asanyarray(test["Price(USD)"])

onehot_2f_model = sklearn.linear_model.LinearRegression()
onehot_2f_model.fit(x_train, y_train)
print(f"Coefficients: {onehot_2f_model.coef_}")
print(f"Intercept: {onehot_2f_model.intercept_: .2f}")

y_pred = onehot_2f_model.predict(x_test)

mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = sklearn.metrics.r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse: .2f}")
print(f"Root Mean Squared error (RMSE): {rmse: .2f}")
print(f"R-squared (R²): {r2: .4f}")

onehot_6f_df = df[["Area", "Room", "Warehouse", "Elevator", "Parking", *encoded.columns, "Price(USD)"]].copy()

train = onehot_6f_df[mask]
test = onehot_6f_df[~mask]

x_train = np.asanyarray(train[["Area", "Room", "Warehouse", "Elevator", "Parking", *encoded.columns]])
y_train = np.asanyarray(train["Price(USD)"])

x_test = np.asanyarray(test[["Area", "Room", "Warehouse", "Elevator", "Parking", *encoded.columns]])
y_test = np.asanyarray(test["Price(USD)"])

onehot_6f_model = sklearn.linear_model.LinearRegression()
onehot_6f_model.fit(x_train, y_train)
print(f"Coefficients: {onehot_6f_model.coef_}")
print(f"Intercept: {onehot_6f_model.intercept_: .2f}")

y_pred = onehot_6f_model.predict(x_test)

mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = sklearn.metrics.r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse: .2f}")
print(f"Root Mean Squared error (RMSE): {rmse: .2f}")
print(f"R-squared (R²): {r2: .4f}")
