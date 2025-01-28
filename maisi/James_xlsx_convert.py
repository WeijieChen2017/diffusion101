import pandas as pd
import os

# Define the path to the directory containing the Excel files
directory_path = "NAC_CTAC_Spacing15"

# Create an empty DataFrame to store the combined data
combined_df = pd.DataFrame()

# Loop through each Excel file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(directory_path, filename)
        
        # Read each workbook into a DataFrame
        df = pd.read_excel(file_path)
        
        # Append the data to the combined DataFrame
        combined_df = combined_df.append(df, ignore_index=True)

# Write the combined DataFrame to a new Excel file
combined_df.to_excel('combined_workbook.xlsx', index=False)