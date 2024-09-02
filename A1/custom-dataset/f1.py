import pandas as pd
from sklearn.metrics import f1_score

# Load the actual labels and predicted labels from the Excel files without headers
actual_df = pd.read_excel('Aug24-Assignment1-Dataset1-test.xlsx', header=None)
predicted_df = pd.read_excel('test_predictions1.xlsx', header=None)

# Assuming the labels are in the first column (index 0) in both files
actual_labels = actual_df.iloc[:, 0]
predicted_labels = predicted_df.iloc[:, 1]

# Calculate the micro-F1 score
micro_f1 = f1_score(actual_labels, predicted_labels, average='micro')

print(f'Micro-F1 Score: {micro_f1:.4f}')