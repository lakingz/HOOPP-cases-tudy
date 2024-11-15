import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import re
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

file_path = 'C:/Users/lakin/OneDrive/Desktop/Ml interview/DA/HOOPP/data/combined_data_train_clean.csv'
df = pd.DataFrame()
df = pd.read_csv(file_path)

# Summary of the dataframe
len(grouped_df)
print("DataFrame Information:")
print(grouped_df.info())  # Provides an overview of column types and non-null values

print("\nDataFrame Head (First 5 rows):")
print(df.head())  # Display the first few rows of the dataset

print("\nMissing Values in DataFrame:")
print(grouped_df.isnull().sum())  # Check for missing values in each column

print("\nUnique Values in DataFrame:")
print(grouped_df.nunique())  # Check for missing values in each column
print(df.notna().sum())  # Check for missing values in each column


def remove_numbers(address):
    return re.sub(r'\d+', '', address)  # Replace all numbers with an empty string

# Apply the function to the 'full address' column

df = df.dropna(subset=['primary_street', 'province', 'infraction_code'])

min_class_count = 100  # Adjust this value as needed
class_counts = df['primary_street'].value_counts()
valid_classes = class_counts[class_counts >= min_class_count].index
df = df[df['primary_street'].isin(valid_classes)]

class_counts = df['infraction_code_description'].value_counts()
valid_classes = class_counts[class_counts >= min_class_count].index
df = df[df['infraction_code_description'].isin(valid_classes)]



grouped_df = df.groupby(
    ['province', 'year', 'month', 'day_of_week', 'hour_of_infraction', 'primary_street', 'infraction_code_description']
).size().reset_index(name='count')

#output_path = 'C:/Users/lakin/OneDrive/Desktop/Ml interview/DA/HOOPP/data/combined_data_train_count_totrain.csv'
#grouped_df.to_csv(output_path, index=False)
file_path = 'C:/Users/lakin/OneDrive/Desktop/Ml interview/DA/HOOPP/data/combined_data_train_count_totrain.csv'
grouped_df = pd.DataFrame()
grouped_df = pd.read_csv(file_path)


grouped_df = grouped_df.dropna(subset=['primary_street'])
len(grouped_df)
len(grouped_df['count'].unique())

# Features: time, location, fine amount
categorical_features = ['province', 'primary_street', 'infraction_code_description']  # List of categorical features
numeric_features = ['year', 'month', 'day_of_week', 'hour_of_infraction']  # List of numeric features
target = 'count'  # Target variable

# Combine categorical and numeric features
features = numeric_features + categorical_features# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(grouped_df[features], grouped_df[target], test_size=0.3, random_state=42)

# Prepare the Pool object for CatBoost (handles categorical features internally)
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
test_pool = Pool(X_test, y_test, cat_features=categorical_features)

# Initialize the CatBoost Regressor
model = CatBoostRegressor(
    iterations=1,          # Number of boosting iterations
    learning_rate=0.02,      # Lower learning rate for better generalization
    depth=8,                 # Depth of trees (can tune this)
 #   l2_leaf_reg=3,
    loss_function='RMSE',     # Regression loss function (Root Mean Squared Error)
    cat_features=categorical_features,
    eval_metric='RMSE',
    custom_metric=['Poisson'],   # Additional metrics to track
    verbose=1
    #    eval_metric='MultiClass',    # Primary metric to optimize
)

params_distribution = {
    'learning_rate': stats.uniform(0.01, 0.1),
    'depth': list(range(3, 10)),
    'l2_leaf_reg': stats.uniform(1, 10),
    'boosting_type': ['Ordered', 'Plain'],
}
grid_search_result = model.randomized_search(
    params_distribution,
    X=X_train,
    y=y_train,
    cv=3,  # Cross-validation folds
    plot=True
)
print("Best parameters:", grid_search_result['params'])

batch_size = 10000  # Set batch size
n_batches = len(X_train) // batch_size
itera=1
i = 0
j = 0 
for j in range(itera):
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(X_train)) 
        X_batch = X_train[start:end]
        y_batch = y_train[start:end]    
        if i+j == 0:  # First batch: No init_model
            model.fit(X_batch, y_batch)
        else:  # Subsequent batches: Use init_model    
            model.fit(X_batch, y_batch, init_model=model)
            
# Get the evaluation results for the training and validation sets
evals_result = model.get_evals_result()

# Print evaluation results
print(evals_result)

# Make predictions on the test set
y_pred = np.round(model.predict(test_pool))
plt.hist(abs(y_test-np.round(y_pred))/y_test)
plt.show()
# Encode string labels to integers

# Calculate F1, Precision, and Recall
f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multi-class
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# If this is binary classification, calculate AUC
if len(np.unique(y_test)) == 2:
    auc = roc_auc_score(y_test, y_pred)
else:
    auc = None  # AUC for multi-class can be computed using 'ovr' (one-vs-rest) approach if needed

# Output the metrics
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
if auc is not None:
    print(f"AUC: {auc}")
else:
    print("AUC is not applicable for multi-class classification.")

train_metrics = model.get_evals_result()

# Plot the metrics (for example, F1 score)
iterations = range(len(train_metrics['validation']))

plt.figure(figsize=(10, 6))
plt.plot(iterations, train_metrics['validation']['F1'], label='F1 Score (Validation)')
plt.plot(iterations, train_metrics['validation']['Precision'], label='Precision (Validation)')
plt.plot(iterations, train_metrics['validation']['Recall'], label='Recall (Validation)')
plt.plot(iterations, train_metrics['validation']['AUC'], label='AUC (Validation)')

plt.xlabel('Iterations')
plt.ylabel('Score')
plt.title('Training Metrics Over Iterations')
plt.legend()
plt.show()
