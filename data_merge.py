import pandas as pd
import os

# Set the folder path where your CSV files are located
folder_path = 'C:/Users/lakin/OneDrive/Desktop/Ml interview/DA/HOOPP/data/data_train/'
output_path = 'C:/Users/lakin/OneDrive/Desktop/Ml interview/DA/HOOPP/data/combined_data_train.csv'
df = pd.DataFrame()
# Get the list of all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Initialize a variable to store the header of the first file
first_header = None

# Initialize an empty dataframe to concatenate into and a list to track mismatched files
merged_df = pd.DataFrame()
mismatch_files = []

# Loop through the files, check headers, and concatenate
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)

    # Get the header of the current file
    current_header = df.columns.tolist()

    # If it's the first file, set its header as the reference
    if first_header is None:
        first_header = current_header  # Set the first header as reference
    elif current_header != first_header:
        # If headers do not match, add the file to the mismatch list
        mismatch_files.append(file)
        print(f"Header mismatch found in file: {file}")
        print(f"Expected: {first_header}")
        print(f"Found: {current_header}")
        continue  # Skip this file and continue with the next

    # Concatenate the dataframe if headers match
    merged_df = pd.concat([merged_df, df], ignore_index=True)

# If there are no mismatched files, export the concatenated dataframe
if len(mismatch_files) == 0:
    # Export the merged dataframe to a CSV file
    merged_df.to_csv(output_path, index=False)

    print(f"Combined CSV file saved as: {output_path}")
else:
    # If there are mismatched files, list them
    print("The following files have mismatched headers and were not merged:")
    for mismatch_file in mismatch_files:
        print(mismatch_file)

# Now we run check based on 'date_of_infraction' column
df = pd.read_csv(output_path)

# Ensure the 'date_of_infraction' column is treated as string for extraction
df['date_of_infraction'] = df['date_of_infraction'].astype(str)

# Extract the year, month, and day from 'date_of_infraction'
df['year'] = df['date_of_infraction'].str[:4].astype(int)
df['month'] = df['date_of_infraction'].str[4:6].astype(int)
df['day'] = df['date_of_infraction'].str[6:8].astype(int)

# Check for years from 2017 to 2020
years_to_check = range(2017, 2021)
# Check if all months (1-12) are present for each year without filtering the dataset
missing_months_by_year = {}
for year in years_to_check:
    months_in_year = df[df['year'] == year]['month'].unique()
    missing_months = [month for month in range(1, 13) if month not in months_in_year]
    if missing_months:
        missing_months_by_year[year] = missing_months

if not missing_months_by_year:
    print("All months from January to December are present for each year between 2017 and 2020.")
else:
    print("Missing months by year:")
    for year, months in missing_months_by_year.items():
        print(f"Year {year}: Missing months {months}")

# Check for large gaps in days for each year and month
for year in years_to_check:
    for month in range(1, 13):
        # Filter data for the current year and month
        month_data = df[(df['year'] == year) & (df['month'] == month)]
        
        if not month_data.empty:
            # Sort the data by day and check for gaps
            days = sorted(month_data['day'].unique())
            gaps = [days[i+1] - days[i] for i in range(len(days)-1) if days[i+1] - days[i] > 1]
            
            if gaps:
                print(f"Year {year}, Month {month}: Large gaps in days: {gaps}")
    print(f"Year {year}: End of the check.")