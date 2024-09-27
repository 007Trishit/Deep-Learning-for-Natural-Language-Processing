import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Load data
train_data = pd.read_excel('augmented-ArithOps_Train.xlsx')
valid_data = pd.read_excel('ArithOps_Validation.xlsx')

# Preprocessing function to replace placeholders
def preprocess_data(df):
    updated_descriptions = []
    updated_expressions = []

    for _, row in df.iterrows():
        description = row['Description']
        input_numbers = row['Input Numbers'].split(' ')  # Assuming numbers are space-separated
        expression = row['Equation']

        if not isinstance(expression, str):
            expression = str(expression)
        
        # Replace placeholders like number0, number1, etc. with actual input numbers
        for idx, num in enumerate(input_numbers):
            description = description.replace(f'number{idx}', num)
            expression = expression.replace(f'number{idx}', num)
# 'Translate the following word problem into an prefix expression: ' +
        updated_descriptions.append(description + row['Question'])
        updated_expressions.append(expression)

    updated_df = pd.DataFrame(columns=['Description', 'Equation'])
    updated_df['Description'] = updated_descriptions
    updated_df['Equation'] = updated_expressions
    return updated_df


# Preprocess the train and validation datasets
train_data = preprocess_data(train_data)
valid_data = preprocess_data(valid_data)

# Save preprocessed data to new Excel files
train_data.to_excel('Updated_Train_Data.xlsx', index=False)
valid_data.to_excel('Updated_Validation_Data.xlsx', index=False)

# Dataset class
class ArithmeticDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        input_text = row['Description']
        target_text = row['Equation']
        
        inputs = self.tokenizer(
            input_text, 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )

        targets = self.tokenizer(
            target_text, 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(), 
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }


# Initialize tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# Create DataLoader
train_dataset = ArithmeticDataset(train_data, tokenizer)
valid_dataset = ArithmeticDataset(valid_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)

# Training settings
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to calculate loss for validation set
def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(data_loader)

# Training loop
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    train_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = evaluate_model(model, valid_loader, device)

    print(f"Train loss: {avg_train_loss}")
    print(f"Validation loss: {avg_val_loss}")


def evaluate_prefix(expressions):
    # Split the expression into tokens
    evals = []
    for expression in expressions:
        tokens = expression.split()

        # Reverse the tokens to process them from right to left
        tokens = tokens[::-1]

        # Create an empty stack
        stack = []

        # Iterate through each token
        for token in tokens:
            if token.isdigit():  # If it's a number, push it onto the stack
                stack.append(int(token))
            else:
                # The token is an operator, pop two operands from the stack
                operand1 = stack.pop()
                operand2 = stack.pop()

                # Perform the operation based on the token
                if token == '+':
                    result = operand1 + operand2
                elif token == '-':
                    result = operand1 - operand2
                elif token == '*':
                    result = operand1 * operand2
                elif token == '/':
                    result = operand1 / operand2

                # Push the result back onto the stack
                stack.append(result)

            # The final result will be the only element left in the stack
        evals.append(stack.pop())
    
    return evals


# Save predictions on validation set
def predict_on_validation(model, data_loader, tokenizer, device):
    model.eval()
    predictions = []
    pred_evals = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
            pred_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            predictions.extend(pred_texts)
            pred_evals.extend(evaluate_prefix(pred_texts))

    return predictions, pred_evals

# Get predictions on validation set
val_predictions, val_evals = predict_on_validation(model, valid_loader, tokenizer, device)

# Add predictions to the validation dataframe and save to Excel
valid_data['Predicted Equation'] = val_predictions
valid_data['Output'] = val_evals
valid_data.to_excel('Validation_Predictions.xlsx', index=False)