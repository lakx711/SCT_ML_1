# House Prices Linear Regression

This project implements a linear regression model to predict house prices using the Ames Housing dataset #SkillCraftTechnology. The model uses three key features:

- **GrLivArea**: Above ground living area (square feet)
- **BedroomAbvGr**: Number of bedrooms above ground
- **TotalBathrooms**: Total number of bathrooms (FullBath + 0.5 * HalfBath)

## Features
- Data loading and feature engineering
- Linear regression model training and evaluation
- Visualization of actual vs. predicted sale prices

## How to Run
1. Install dependencies:
   ```
   pip install pandas scikit-learn matplotlib
   ```
2. Run the script:
   ```
   python linear_regression_house_prices.py
   ```
3. The script will print model coefficients, R² score, and save a plot as `actual_vs_predicted.png`.

## Files
- `train.csv`: Training data
- `linear_regression_house_prices.py`: Main script
- `actual_vs_predicted.png`: Output plot

## Author
Lakshit Soni
