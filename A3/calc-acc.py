import pandas as pd

# Load the actual labels and predicted labels from the Excel files without headers
actual_df = pd.read_excel('ArithOps_Validation.xlsx')
predicted_df = pd.read_excel('Validation_Predictions.xlsx')

actuals = actual_df['Output']
preds = predicted_df['Output']


correct_predictions = 0
total_predictions = len(actuals)

for actual, pred in zip(actuals, preds):

    # Compare evaluated prediction with actual output
    if pred == actual:
        correct_predictions += 1

accuracy_percentage = (correct_predictions / total_predictions) * 100

print(f"Accuracy Percentage: {accuracy_percentage}%")
