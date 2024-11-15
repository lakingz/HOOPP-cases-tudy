import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from scipy import stats
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Load and preprocess data
file_path = 'C:/Users/lakin/OneDrive/Desktop/Ml interview/DA/HOOPP/data/combined_data_train_count_totrain.csv'
grouped_df = pd.read_csv(file_path)
grouped_df_filter = grouped_df
len(grouped_df_filter)
# Removing numbers in 'primary_street'

# Encode categorical features
one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop first to avoid multicollinearity
province_encoded = one_hot_encoder.fit_transform(grouped_df_filter[['province']])

# Convert One-Hot Encoding output to DataFrame and add column names
province_encoded_df = pd.DataFrame(province_encoded, columns=one_hot_encoder.get_feature_names_out(['province']))

# Step 2: Frequency Encoding for 'primary_street' and 'infraction_code_description'
for col in ['primary_street', 'infraction_code_description']:
    freq = grouped_df_filter[col].value_counts() / len(grouped_df_filter)  # Calculate frequency of each category
    grouped_df_filter[col + '_freq'] = grouped_df_filter[col].map(freq)    # Map frequency to the column and create new feature

# Step 3: Combine Encoded Features with Original DataFrame
# Drop original 'province' column (since we have it as one-hot encoded columns)
df = grouped_df_filter.drop(['province'], axis=1)

# Concatenate original DataFrame with encoded columns

grouped_df_filter = grouped_df_filter.reset_index(drop=True)
province_encoded_df = province_encoded_df.reset_index(drop=True)
df_encoded = pd.concat([grouped_df_filter, province_encoded_df], axis=1)
df_encoded.isna().sum()
# Separate features and target
numeric_features = ['year', 'month', 'day_of_week', 'hour_of_infraction']
# Add frequency-encoded columns for high-cardinality categorical features
frequency_encoded_features = ['primary_street_freq', 'infraction_code_description_freq']
# Add one-hot encoded columns for 'province'
one_hot_encoded_features = province_encoded_df.columns.tolist()  # Columns from the one-hot encoding
features = numeric_features + frequency_encoded_features + one_hot_encoded_features

X = df_encoded[features]
y = df_encoded['count']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize XGBoost regressor
eval_set = [(X_train, y_train), (X_test, y_test)]
model = XGBRegressor(
    n_estimators=100,            # Equivalent to iterations
    learning_rate=0.1,          # Similar to CatBoost
    max_depth=8,                 # Depth of trees
    reg_lambda=3,                # Equivalent to l2_leaf_reg
    objective='reg:tweedie', 
    tweedie_variance_power=1.5,                 # Metric to monitor; can also use "rmse", "logloss", etc.
    early_stopping_rounds=50,          # Stops if no improvement for 50 rounds
    verbose=50,            
    eval_metric=["mae", "rmse","tweedie-nloglik@1.5","poisson-nloglik"] 
)

model.fit(X_train, y_train,eval_set=eval_set) 

model = XGBRegressor()
model.load_model("xgboost_model.json")
model.fit(X_train, y_train,eval_set=eval_set, xgb_model=model)
model.save_model("xgboost_model.json")
# Predictions
y_pred = np.round(model.predict(X_test))

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)  # RMSE
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")

abserror = abs(y_test-np.round(y_pred)).value_counts()
abserror_df = abserror.reset_index(name='Counts')  # Converts index to a column
abserror_df.columns = ['Value', 'Count']  # Rename columns
# Calculate cumulative distribution
abserror_df['Cumulative Count'] = abserror_df['Count'].cumsum()
abserror_df['Cumulative Percentage'] = abserror_df['Cumulative Count'] / abserror_df['Count'].sum() * 100

# Plot the quantile plot
plt.figure(figsize=(8, 6))
plt.plot(abserror_df['Value'], abserror_df['Cumulative Percentage'], marker='o')
plt.xlabel('Absolute error')
plt.ylabel('Cumulative Percentage (%)')
plt.title('Quantile Plot of Value Counts')
plt.grid(True)
plt.savefig("absolute_error_Quantile.png", format='png', dpi=300)  # You can also use 'jpg' or 'pdf' formats
plt.show()


evals_result = model.evals_result()

# Plot MAE for training and validation
plt.plot(evals_result['validation_0']['mae'], label='Train mae')
plt.plot(evals_result['validation_1']['mae'], label='Validation mae')
plt.xlabel('Iterations')
plt.ylabel('mae')
plt.title('mae During Training')
plt.legend()
plt.savefig("mae During Training.png", format='png', dpi=300)  # You can also use 'jpg' or 'pdf' formats
plt.show()


#
#calibration curve
#
from scipy.stats import binned_statistic

try:
    fig, ax = plt.subplots(figsize=(10, 8))

    # For Training Set
    y_pred_train = model.predict(X_train)
    bin_means_train, bin_edges_train, _ = binned_statistic(y_pred_train, y_train, statistic='mean', bins=10)
    bin_centers_train = (bin_edges_train[1:] + bin_edges_train[:-1]) / 2
    ax.plot(bin_centers_train, bin_means_train, marker='o', linewidth=1, label='Train')

    # For Test Set
    y_pred_test = model.predict(X_test)
    bin_means_test, bin_edges_test, _ = binned_statistic(y_pred_test, y_test, statistic='mean', bins=10)
    bin_centers_test = (bin_edges_test[1:] + bin_edges_test[:-1]) / 2
    ax.plot(bin_centers_test, bin_means_test, marker='o', linewidth=1, label='Test')

    # Perfect calibration line
    ax.plot([min(y_pred_test), max(y_pred_test)], [min(y_pred_test), max(y_pred_test)], linestyle='--', color='gray', label='Perfect Calibration')
    
    ax.set_xlabel('Predicted Count')
    ax.set_ylabel('Mean Actual Count')
    ax.set_title('Calibration Curve for Count Prediction')
    ax.legend()
    
    fig.savefig('Calibration_Curve_Count_Prediction.png')  # Save the figure
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    plt.close('all')


importances = model.feature_importances_
feature_names = X.columns if isinstance(X, pd.DataFrame) else np.arange(X.shape[1])
feature_names
# Sorting the features based on importance
indices = np.argsort(importances)[::-1]
indices
try:
    # Creating the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(range(len(importances[0:10])), importances[indices[0:10]], color='b', align='center')
    ax.set_xticks(range(len(importances[0:10])))
    ax.set_xticklabels(feature_names[indices[0:10]], rotation=45, ha='right', fontsize=10)
    ax.set_title('Top 10 Feature Importances')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance')
    fig.subplots_adjust(bottom=0.3)
    fig.savefig('Top 10 Feature Importances.png')  # Save the figure
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    plt.close('all')

try:
    for i in range(3):  # Loop through first three trees
        xgb.plot_tree(model, num_trees=i, rankdir='LR')
        fig = plt.gcf()  # Get current figure
        fig.set_size_inches(30, 20)  # Resize figure for better readability
        fig.savefig(f'tree_{i + 1}.png')  # Save each tree as a separate file
        plt.clf()  # Clear the current figure
except Exception as e:
    print(f"An error occurred with tree plotting: {e}")
finally:
    plt.close('all')
