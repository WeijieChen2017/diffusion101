import pandas as pd

# Define the path to the Excel file
file_path = "NAC_CTAC_Spacing15/NAC_CTAC_Spacing15.xlsx"

# Read the Excel file and get all sheet names
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names

# Create an empty list to store the DataFrames
dataframes = []

# Loop through each sheet and read it into a DataFrame
for sheet_name in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    dataframes.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Write the combined DataFrame to a new Excel file
combined_df.to_excel('combined_workbook.xlsx', index=False)