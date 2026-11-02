# Tehran House Price Prediction: A Comparative Study of Feature Encoding

## Description

This project explores and compares different approaches to predicting house prices in Tehran using real-world housing data.

It investigates the impact of feature selection and categorical encoding techniques on regression performance, focusing on One-Hot Encoding and Score-Based Encoding. Multiple regression models are implemented and evaluated, including linear and logarithmic regression.

The main objective is to examine how different feature representations and model specifications affect the accuracy and performance of house price prediction.

## Features
- **Data Analysis** : Exploring and analyzing Tehran housing market data.
- **One-Hot Encoding** : Representing categorical neighborhood data using dummy variables.
- **Score-Based Encoding** : Assigning numerical scores to neighborhoods based on their relative property values.
- **Feature Comparison** : Comparing models with 2 and 6 input features.
- **Linear Regression** : Implementing linear regression with different feature configurations.
- **Logarithmic Regression:** : Exploring logarithmic relationships between housing features and prices.
- **Model Evaluation** : Evaluating model performance using R², MAE, and RMSE.
- **Data Visualization** : Visualizing housing data, regression results, and prediction errors.

## Technologies

- **Python** : Main programming language.
- **NumPy** : Numerical computations and array operations.
- **Pandas** : Data loading, cleaning, preprocessing, and manipulation.
- **Matplotlib** : Data visualization and plotting regression results.
- **Scikit-learn** :  Implementing linear regression models and evaluating their performance.
- **SciPy** : Nonlinear curve fitting using curve_fit for logarithmic regression.

## Project Structure

```test
second_main_proj/
├── regression.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

On Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the project

```bash
python first_ml.py
```

## Machine Learning

This project implements five regression models to predict residential property prices in Tehran. The models compare two address encoding techniques (One-Hot Encoding and Score-Based Encoding), different feature sets, and logarithmic regression.

### Implemented Models

1. **Linear Regression – One-Hot Encoding (2 Features):** Uses Area and One-Hot Encoded Address features.
2. **Linear Regression – One-Hot Encoding (6 Features):** Uses Area, One-Hot Encoded Address, Room, Elevator, Warehouse, and Parking.
3. **Linear Regression – Score-Based Encoding (2 Features):** Uses Area and Score to represent property location.
4. **Linear Regression – Score-Based Encoding (6 Features):** Uses Area, Score, Room, Elevator, Warehouse, and Parking.
5. **Logarithmic Regression:** Uses a logarithmic function of Area and Score to model their relationship with house prices.

### Model Evaluation

The models are evaluated using the following regression metrics:

- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R-squared (R²)**

### Model Results

| Model | MSE (USD²)| RMSE(USD) |R²|
|---|---:|---:|---:|
| Linear Regression – One-Hot (2 Features)| 23,874,944,673.65 | 154,515.19 | 0.6745 |
| Linear Regression – One-Hot (6 Features) | 24,127,102,724.51 | 155,329.01 | 0.6710 |
| Linear Regression – Score-Based (2 Features) | 34,313,513,584.95 | 185,239.07 | 0.5204 |
| Linear Regression – Score-Based (6 Features) | 33,720,166,432.72 | 183,630.52 | 0.5287 |
| Logarithmic Regression | 37,967,282,544.58 | 194,851.95 | 0.5117 |

## License

This project is available under the MIT License.