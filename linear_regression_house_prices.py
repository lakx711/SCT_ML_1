import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load the data
train = pd.read_csv('train.csv')

# Feature engineering: total bathrooms = FullBath + 0.5 * HalfBath
train['TotalBathrooms'] = train['FullBath'] + 0.5 * train['HalfBath']

# Select features and target
features = ['GrLivArea', 'BedroomAbvGr', 'TotalBathrooms']
X = train[features]
y = train['SalePrice']

# Split the data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Print coefficients
print('Intercept:', model.intercept_)
print('Coefficients:')
for feature, coef in zip(features, model.coef_):
    print(f'  {feature}: {coef}')

# Evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'R^2 score on test set: {r2:.4f}')

# Visualization: Actual vs. Predicted SalePrice
y_test_np = np.array(y_test)
y_pred_np = np.array(y_pred)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_np, y_pred_np, alpha=0.6)
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Actual vs. Predicted SalePrice')
plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--')  # Diagonal line
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show() 