import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re

file_path = 'C:/Users/lakin/OneDrive/Desktop/Ml interview/DA/HOOPP/data/combined_data_train.csv'
df = pd.DataFrame()
df = pd.read_csv(file_path)
len(df)
# Display the first few rows
print(df.head())

# Handle missing values by checking for any NaNs
df.isnull().sum()
df = df.dropna(subset=['province', 'infraction_code'])
#
# Handle Missing Values and Date Parsing
#
# Convert 'date_of_infraction' to datetime format
df['date_of_infraction'] = pd.to_datetime(df['date_of_infraction'], format='%Y%m%d', errors='coerce')
# Convert the 'time_of_infraction' column to strings and pad with leading zeros to ensure 'HHMM' format
df['time_of_infraction'] = df['time_of_infraction'].fillna('0').astype(int).astype(str).apply(lambda x: x.zfill(4))

# Create a new column for the actual time by converting the padded string 'HHMM' to 'HH:MM'
def convert_to_time(time_str):
    try:
        return datetime.strptime(time_str, '%H%M').time()  # Convert 'HHMM' format to time object
    except ValueError:
        return None  # Handle any conversion errors
    
df['infraction_time'] = df['time_of_infraction'].apply(convert_to_time)

# Combine 'date_of_infraction' and 'infraction_time' into a full datetime column
df['datetime_of_infraction'] = df.apply(lambda row: datetime.combine(row['date_of_infraction'], row['infraction_time']) if pd.notnull(row['infraction_time']) else None, axis=1)

# Display the first few rows with the new combined 'datetime_of_infraction'
print(df[['date_of_infraction', 'infraction_time', 'datetime_of_infraction']].head())

# Check the data type of the new 'datetime_of_infraction' column
print(df['datetime_of_infraction'].dtype)
df = df.dropna(subset=['datetime_of_infraction'])

# Check the ‘infraction_code’ and 'infraction_description' column
print(len(df['infraction_code'].unique()))
print(len(df['infraction_description'].unique()))
print(len((df['infraction_code'].astype(str)+df['infraction_description'].astype(str)).unique()))

df['infraction_code_description'] = df[['infraction_code','infraction_description']].fillna('').astype(str).agg(' '.join, axis=1).str.strip()

#
#Concatenate Location Fields
#
# Combine the location columns into one full address


def remove_numbers(address):
    return re.sub(r'\d+', '', address)  # Replace all numbers with an empty string
df = df.dropna(subset=['location2'])
df['primary_street'] = df['location2']
df = df.dropna(subset=['primary_street'])

df['primary_street'] = df['primary_street'].str.replace(r'[^a-zA-Z]', ' ', regex=True).str.upper()

# Step 2: Remove any extra spaces (more than one space in a row)
df['primary_street'] = df['primary_street'].str.replace(r'\s+', ' ', regex=True).str.strip()

print(len(df['primary_street'].unique()))

# create features for 'year', 'month', 'day_of_week', and 'hour_of_infraction'
df['year'] = df['datetime_of_infraction'].dt.year
df['month'] = df['datetime_of_infraction'].dt.month
df['day_of_week'] = df['datetime_of_infraction'].dt.weekday
df['hour_of_infraction'] = df['datetime_of_infraction'].dt.hour

#
# Save the cleaned data to a CSV file
#
output_path = 'C:/Users/lakin/OneDrive/Desktop/Ml interview/DA/HOOPP/data/combined_data_train_clean.csv'
df.to_csv(output_path, index=False)
file_path = 'C:/Users/lakin/OneDrive/Desktop/Ml interview/DA/HOOPP/data/combined_data_train_clean.csv'
df = pd.DataFrame()
df = pd.read_csv(file_path)

len(df)
print("DataFrame Information:")
print(df.info())  # Provides an overview of column types and non-null values

print("\nDataFrame Head (First 5 rows):")
print(df.head())  # Display the first few rows of the dataset

print("\nMissing Values in DataFrame:")
print(df.isnull().sum())  # Check for missing values in each column

print("\nUnique Values in DataFrame:")
print(df.nunique())  # Check for missing values in each column
print(df.notna().sum())  # Check for missing values in each column

#
# some basic EDA plots
#
# Plot the number of infractions per year
infractions_per_year = df.groupby('year').size()
infractions_per_year.plot(kind='bar', color='skyblue', figsize=(10, 5))
plt.title("Parking Infractions by Year")
plt.xlabel("Year")
plt.ylabel("Number of Infractions")
plt.show()

# Plot infractions per month (for all years combined)
infractions_per_month = df.groupby('month').size()
infractions_per_month.plot(kind='bar', color='green', figsize=(10, 5))
plt.title("Parking Infractions by Month")
plt.xlabel("Month")
plt.ylabel("Number of Infractions")
plt.show()

# Plot infractions per day_of_week (for all years combined)
infractions_per_day_of_week = df.groupby('day_of_week').size()
infractions_per_day_of_week.plot(kind='bar', color='green', figsize=(10, 5))
plt.title("Parking Infractions by day_of_week")
plt.xlabel("day_of_week")
plt.ylabel("Number of Infractions")
plt.show()

# Find the most common infraction types
top_infractions = df['infraction_description'].value_counts().head(10)
top_infractions.plot(kind='bar', color='orange', figsize=(6, 3))
plt.title("Top 10 Most Common Parking Infractions")
plt.xlabel("Infraction Description")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Plot the distribution of fines
plt.figure(figsize=(10, 5))
sns.histplot(df['set_fine_amount'], kde=False, color='purple', bins=30)
plt.title("Distribution of Parking Fines")
plt.xlabel("Fine Amount")
plt.ylabel("Frequency")
plt.show()

# Find the most common violation locations (top 10)
top_locations = df['full_address'].value_counts().head(10)
top_locations.plot(kind='bar', color='red', figsize=(10, 5))
plt.title("Top 10 Violation Locations")
plt.xlabel("Location")
plt.ylabel("Number of Infractions")
plt.xticks(rotation=45)
plt.show()

# Plot the number of infractions by hour of the day
infractions_by_hour = df.groupby('hour_of_infraction').size()
infractions_by_hour.plot(kind='bar', color='blue', figsize=(10, 5))
plt.title("Parking Infractions by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Number of Infractions")
plt.show()
