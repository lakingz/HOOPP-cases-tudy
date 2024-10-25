import pandas as pd
import catboost as cb
from datetime import datetime
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

file_path = 'C:/Users/lakin/OneDrive/Desktop/Ml interview/DA\HOOPP/data/combined_data_train_clean.csv'
df = pd.DataFrame()
df = pd.read_csv(file_path)

# Summary of the dataframe
print("DataFrame Information:")
print(df.info())  # Provides an overview of column types and non-null values

print("\nDataFrame Head (First 5 rows):")
print(df.head())  # Display the first few rows of the dataset

print("\nMissing Values in DataFrame:")
print(df.isnull().sum())  # Check for missing values in each column

df = df.dropna(subset=['full_address', 'province'])
len(df['full_address'].unique())

# Features: time, location, fine amount
X = df[['hour_of_infraction', 'full_address', 'month']]
y = df['set_fine_amount'] # Binary target for prohibited parking

categorical_features = ['infraction_description', 'full_address']  # List of categorical features
numeric_features = ['year', 'month', 'day_of_week', 'hour_of_infraction']  # List of numeric features
target = 'set_fine_amount'  # Target variable

# Combine categorical and numeric features
features = numeric_features + categorical_features# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=42)

# Prepare the Pool object for CatBoost (handles categorical features internally)
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
test_pool = Pool(X_test, y_test, cat_features=categorical_features)

# Initialize the CatBoost Regressor
model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=100)

# Train the model
model.fit(train_pool)

# Make predictions on the test set
y_pred = model.predict(test_pool)

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
