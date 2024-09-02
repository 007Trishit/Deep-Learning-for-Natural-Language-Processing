import pandas as pd

# Load the original and validation datasets
original_df = pd.read_excel('custom-dataset/Aug24-Assignmen1-Dataset1.xlsx')
validation_df = pd.read_excel('custom-dataset/Aug24-Assignment1-Validation-Dataset1.xlsx')

# Remove common data based on 'reviewText'
filtered_df = original_df[~original_df['reviewText'].isin(validation_df['reviewText'])]

# Optionally, save the filtered dataset to a new Excel file
filtered_df.to_excel('custom-dataset/Aug24-Assignment1-Train-Dataset1.xlsx', index=False)

print(f"Original dataset rows: {len(original_df)}")
print(f"Filtered dataset rows: {len(filtered_df)}")