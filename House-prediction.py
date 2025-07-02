#  Step 1: Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  For loading dataset and building the model
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#  Step 2: Load the California housing dataset
# This dataset contains housing data like income, rooms, population, etc.
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['HousePrice'] = data.target  # Target is the median house value

#  Preview the first few rows of data
print("Sample data:")
print(df.head())

#  Step 3: Check for missing values (if any)
print("\nMissing values in each column:")
print(df.isnull().sum())  # Great! Should be 0 for this dataset

#  Step 4: Explore the data with visualizations
# Let’s see how different features relate to the house price

# Heatmap to visualize correlations
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("🔗 Correlation Heatmap")
plt.show()

# Scatter plot: Median Income vs House Price
plt.figure(figsize=(8, 6))
sns.scatterplot(x='MedInc', y='HousePrice', data=df, color='green', alpha=0.6)
plt.title("Income vs House Price")
plt.xlabel("Median Income (in 10,000s)")
plt.ylabel("House Price (in 100,000s)")
plt.grid(True)
plt.show()

# ----------------------------------------
#  Step 5: Preparing the data for modeling
# ----------------------------------------
X = df.drop('HousePrice', axis=1)  # Input features
y = df['HousePrice']               # Target feature

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------
#  Step 6: Train a Linear Regression model
# ----------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------------------
#  Step 7: Make predictions on test data
# ----------------------------------------
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score (closer to 1 is better): {r2:.2f}")

# ----------------------------------------
#  Step 8: Visualize predictions vs actual values
# ----------------------------------------

# Plot: Actual vs Predicted Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.grid(True)
plt.show()

# Plot: Residuals (Prediction Errors)
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='red')
plt.title("Residuals Distribution (Prediction Errors)")
plt.xlabel("Error (Actual - Predicted)")
plt.grid(True)
plt.show()
