# Used Car Price Prediction Model
# ANA680 Final Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Data Loading and Initial Exploration
print("1. Loading and exploring the Used Cars dataset")
df = pd.read_csv('Used_Cars.csv')

# Display basic information
print(f"\nDataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values by column:")
print(df.isnull().sum())

# 2. Data Cleaning
print("\n2. Data Cleaning")

# Function to clean price values
def clean_price(price_str):
    if pd.isna(price_str):
        return np.nan
    try:
        # Remove non-numeric characters and convert to float
        return float(re.sub(r'[^\d.]', '', str(price_str)))
    except:
        return np.nan

# Function to clean mileage values
def clean_mileage(mileage_str):
    if pd.isna(mileage_str):
        return np.nan
    try:
        # Extract numeric values and convert to float
        return float(re.sub(r'[^\d.]', '', str(mileage_str)))
    except:
        return np.nan

# Clean price and mileage columns
df['price_cleaned'] = df['price'].apply(clean_price)
df['mileage_cleaned'] = df['milage'].apply(clean_mileage)

# Drop rows with missing or zero prices
df = df[df['price_cleaned'] > 0].reset_index(drop=True)

# Handle outliers in price
price_q1 = df['price_cleaned'].quantile(0.01)
price_q3 = df['price_cleaned'].quantile(0.99)
df = df[(df['price_cleaned'] >= price_q1) & (df['price_cleaned'] <= price_q3)]

# Handle outliers in mileage
mileage_q1 = df['mileage_cleaned'].quantile(0.01)
mileage_q3 = df['mileage_cleaned'].quantile(0.99)
df = df[(df['mileage_cleaned'] >= mileage_q1) & (df['mileage_cleaned'] <= mileage_q3)]

# Fill missing values for categorical columns
df['fuel_type'] = df['fuel_type'].fillna('Unknown')
df['accident'] = df['accident'].fillna('Unknown')
df['clean_title'] = df['clean_title'].fillna('Unknown')

# Create binary feature for accident
df['has_accident'] = df['accident'].apply(lambda x: 0 if 'None' in str(x) else 1)

# Simplify engine feature by extracting engine size
def extract_engine_size(engine_str):
    if pd.isna(engine_str):
        return np.nan
    try:
        # Find patterns like 3.5L, 2.0, 5.7 Liter, etc.
        match = re.search(r'(\d+\.\d+)(?:L| Liter)', str(engine_str))
        if match:
            return float(match.group(1))
        else:
            return np.nan
    except:
        return np.nan

df['engine_size'] = df['engine'].apply(extract_engine_size)

# Extract transmission type (Automatic or Manual)
def simplify_transmission(trans_str):
    if pd.isna(trans_str):
        return 'Unknown'
    trans_lower = str(trans_str).lower()
    if 'a/t' in trans_lower or 'auto' in trans_lower:
        return 'Automatic'
    elif 'm/t' in trans_lower or 'manual' in trans_lower:
        return 'Manual'
    else:
        return 'Other'

df['transmission_type'] = df['transmission'].apply(simplify_transmission)

# Calculate car age
current_year = 2025  # Current year as of analysis
df['car_age'] = current_year - df['model_year']

print("\nAfter cleaning, dataset shape:", df.shape)
print("\nCleaned data sample:")
print(df[['brand', 'model', 'model_year', 'mileage_cleaned', 'price_cleaned', 'engine_size', 'transmission_type', 'car_age']].head())

# 3. Exploratory Data Analysis (EDA)
print("\n3. Exploratory Data Analysis")

# Price distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['price_cleaned'], kde=True)
plt.title('Distribution of Car Prices')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.savefig('price_distribution.png')
plt.close()

# Mileage vs Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='mileage_cleaned', y='price_cleaned', data=df, alpha=0.5)
plt.title('Price vs Mileage')
plt.xlabel('Mileage')
plt.ylabel('Price ($)')
plt.savefig('price_vs_mileage.png')
plt.close()

# Car Age vs Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='car_age', y='price_cleaned', data=df, alpha=0.5)
plt.title('Price vs Car Age')
plt.xlabel('Car Age (years)')
plt.ylabel('Price ($)')
plt.savefig('price_vs_age.png')
plt.close()

# Brand vs Price
top_brands = df['brand'].value_counts().head(10).index
df_top_brands = df[df['brand'].isin(top_brands)]

plt.figure(figsize=(12, 8))
sns.boxplot(x='brand', y='price_cleaned', data=df_top_brands)
plt.title('Price by Top 10 Brands')
plt.xlabel('Brand')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)
plt.savefig('price_by_brand.png')
plt.close()

# Transmission Type vs Price
plt.figure(figsize=(10, 6))
sns.boxplot(x='transmission_type', y='price_cleaned', data=df)
plt.title('Price by Transmission Type')
plt.xlabel('Transmission Type')
plt.ylabel('Price ($)')
plt.savefig('price_by_transmission.png')
plt.close()

# Has Accident vs Price
plt.figure(figsize=(10, 6))
sns.boxplot(x='has_accident', y='price_cleaned', data=df)
plt.title('Price by Accident History')
plt.xlabel('Has Accident (1 = Yes, 0 = No)')
plt.ylabel('Price ($)')
plt.savefig('price_by_accident.png')
plt.close()

# Correlation matrix
numerical_cols = ['price_cleaned', 'mileage_cleaned', 'model_year', 'car_age', 'engine_size', 'has_accident']
df_corr = df[numerical_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()

print("\nCorrelation with price:")
print(df_corr['price_cleaned'].sort_values(ascending=False))

# 4. Feature Engineering and Selection
print("\n4. Feature Engineering and Selection")

# Select features for modeling
features = df[['brand', 'model_year', 'mileage_cleaned', 'fuel_type', 
              'engine_size', 'transmission_type', 'ext_col', 'has_accident', 'car_age']]
target = df['price_cleaned']

# Handle missing values
features['engine_size'] = features['engine_size'].fillna(features['engine_size'].median())

# Print feature information
print("\nSelected features:")
for col in features.columns:
    if features[col].dtype == 'object':
        print(f"{col}: Categorical with {features[col].nunique()} unique values")
    else:
        print(f"{col}: Numerical with range {features[col].min():.2f} to {features[col].max():.2f}")

# 5. Model Training and Evaluation
print("\n5. Model Training and Evaluation")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Define preprocessor
numerical_features = ['model_year', 'mileage_cleaned', 'engine_size', 'car_age']
categorical_features = ['brand', 'fuel_type', 'transmission_type', 'ext_col']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ],
    remainder='passthrough'
)

# Define and train models
models = {
    'Linear Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ]),
    
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
    ])
}

# Function to evaluate models
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{name} Performance:")
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    return {'model': model, 'mae': mae, 'rmse': rmse, 'r2': r2}

# Train and evaluate all models
results = {}
for name, model in models.items():
    results[name] = evaluate_model(name, model, X_train, X_test, y_train, y_test)

# Find the best model
best_model_name = min(results, key=lambda k: results[k]['rmse'])
best_model = results[best_model_name]['model']

print(f"\nBest performing model: {best_model_name}")
print(f"RMSE: ${results[best_model_name]['rmse']:.2f}")

# 6. Model Optimization
print("\n6. Model Optimization")

if best_model_name == 'Random Forest':
    # Hyperparameter tuning for Random Forest
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(best_model, param_grid, cv=3, scoring='neg_root_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    
    # Evaluate optimized model
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nOptimized {best_model_name} Performance:")
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R²: {r2:.4f}")

elif best_model_name == 'XGBoost':
    # Hyperparameter tuning for XGBoost
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.01, 0.1, 0.2]
    }
    
    grid_search = GridSearchCV(best_model, param_grid, cv=3, scoring='neg_root_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    
    # Evaluate optimized model
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nOptimized {best_model_name} Performance:")
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R²: {r2:.4f}")

# 7. Feature Importance Analysis
print("\n7. Feature Importance Analysis")

if best_model_name == 'Random Forest' or best_model_name == 'XGBoost':
    # Get feature names after preprocessing
    categorical_ohe = best_model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
    cat_features = categorical_ohe.get_feature_names_out(categorical_features)
    feature_names = list(numerical_features) + list(cat_features) + ['has_accident']
    
    # Get feature importances
    importances = best_model.named_steps['regressor'].feature_importances_
    
    # Create dataframe for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names[:len(importances)],
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
    plt.title(f'Top 15 Feature Importances ({best_model_name})')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("\nTop 10 important features:")
    print(importance_df.head(10))

# 8. Saving the Model
print("\n8. Saving the Model")

# Save the final model
model_filename = 'car_price_prediction_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)

print(f"Model saved to {model_filename}")

# Save the training process and important information as a summary
summary = {
    'dataset_shape': df.shape,
    'features_used': list(features.columns),
    'best_model': best_model_name,
    'metrics': {
        'mae': results[best_model_name]['mae'],
        'rmse': results[best_model_name]['rmse'],
        'r2': results[best_model_name]['r2']
    }
}

summary_filename = 'model_summary.pkl'
with open(summary_filename, 'wb') as file:
    pickle.dump(summary, file)

print(f"Model summary saved to {summary_filename}")
print("\nModel development complete!")

# 9. Sample Prediction Function
def predict_car_price(model, brand, year, mileage, fuel_type, engine_size, transmission, ext_color, has_accident):
    """
    Function to make predictions with our trained model
    """
    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'brand': [brand],
        'model_year': [year],
        'mileage_cleaned': [mileage],
        'fuel_type': [fuel_type],
        'engine_size': [engine_size],
        'transmission_type': [transmission],
        'ext_col': [ext_color],
        'has_accident': [has_accident],
        'car_age': [2025 - year]
    })
    
    # Make prediction
    predicted_price = model.predict(input_data)[0]
    
    return predicted_price

# Example usage:
print("\n9. Sample Prediction:")
sample_car = {
    'brand': 'Toyota',
    'year': 2018,
    'mileage': 50000,
    'fuel_type': 'Gasoline',
    'engine_size': 2.5,
    'transmission': 'Automatic',
    'ext_color': 'White',
    'has_accident': 0
}

predicted_price = predict_car_price(
    best_model, 
    sample_car['brand'], 
    sample_car['year'],
    sample_car['mileage'],
    sample_car['fuel_type'],
    sample_car['engine_size'],
    sample_car['transmission'],
    sample_car['ext_color'],
    sample_car['has_accident']
)

print(f"Predicted price for a {sample_car['year']} {sample_car['brand']} with {sample_car['mileage']} miles: ${predicted_price:.2f}")
