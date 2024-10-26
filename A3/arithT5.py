#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch.nn.functional as F
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
import re

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Load the train and validation datasets from Excel files
train_file_path = '../ArithOps_Train.xlsx'  # Replace with your train file path
validation_file_path = '../ArithOps_Validation.xlsx'  # Replace with your validation file path


# In[4]:


train_df = pd.read_excel(train_file_path)
validation_df = pd.read_excel(validation_file_path)

# Preprocess data: Remove rows with null values in any column
train_df = train_df.dropna()
validation_df = validation_df.dropna()


# In[5]:





# In[6]:


# Function to prepare data (concatenate description and question, and keep equation as target)
def prepare_data(df):
    inputs = []
    targets = []
    
    for _, row in df.iterrows():
        description = row['Description']
        question = row['Question']
        equation = row['Equation']

        # Concatenate description and question as input
        inputs.append(f"{description} {question}")
        targets.append(equation)
    
    return inputs, targets


# In[7]:


# Prepare the train and validation datasets
train_inputs, train_targets = prepare_data(train_df)
val_inputs, val_targets = prepare_data(validation_df)

# Tokenize the inputs and targets
def tokenize_data(inputs, targets, tokenizer, max_length=128):
    input_encodings = tokenizer(inputs, truncation=True, padding=True, max_length=max_length)
    target_encodings = tokenizer(targets, truncation=True, padding=True, max_length=max_length)
    
    return input_encodings, target_encodings

# Tokenize train and validation data
train_input_encodings, train_target_encodings = tokenize_data(train_inputs, train_targets, tokenizer)
val_input_encodings, val_target_encodings = tokenize_data(val_inputs, val_targets, tokenizer)


# In[14]:


# Custom Dataset class for PyTorch
class ArithmeticDataset(Dataset):
    def __init__(self, input_encodings, target_encodings):
        self.input_encodings = input_encodings
        self.target_encodings = target_encodings

    def __len__(self):
        return len(self.input_encodings['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.input_encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.target_encodings['input_ids'][idx]),
        }

# Create datasets
train_dataset = ArithmeticDataset(train_input_encodings, train_target_encodings)
val_dataset = ArithmeticDataset(val_input_encodings, val_target_encodings)

# DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)


# In[15]:


# model_path = './best_t5_arithmetic_model'  # Path to the trained model
# tokenizer = T5Tokenizer.from_pretrained(model_path)
# model = T5ForConditionalGeneration.from_pretrained(model_path)
# Set up optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


# In[16]:


from tqdm import tqdm
# Function to replace number placeholders with actual numbers
def replace_numbers_in_equation(equation, input_numbers):
    # Split input numbers into a list
    numbers = input_numbers.split()
    
    # Replace number0, number1, ... with corresponding input numbers
    for i, number in enumerate(numbers):
        equation = equation.replace(f"number{i}", number)
    
    return equation


import operator

# Function to evaluate prefix notation
def evaluate_prefix(expression):
    # Split the expression into tokens
    tokens = expression.split()
    
    # Stack to hold operands
    stack = []
    
    # Define operator functions
    ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv
    }
    
    # Traverse the tokens in reverse (right-to-left)
    for token in reversed(tokens):
        if token in ops:
            # Pop two operands from the stack for the operation
            operand1 = stack.pop()
            operand2 = stack.pop()
            
            # Apply the operator and push the result back to the stack
            result = ops[token](operand1, operand2)
            stack.append(result)
        else:
            # If it's a number, push it to the stack
            stack.append(float(token))  # Convert numbers to float
    
    # The final result will be the only item left in the stack
    return stack[0]

# Evaluate the equation by replacing placeholders and using eval
def evaluate_equation(equation):
    try:
        # Safely evaluate the arithmetic expression
        return evaluate_prefix(equation)
    except Exception as e:
        print(f"Error evaluating equation: {equation}. Error: {e}")
        return None

# Accuracy calculation function
def calculate_accuracy(predicted_equations, df):
    correct_predictions = 0
    total_predictions = 0

    for i, row in df.iterrows():
        input_numbers = row['Input Numbers']
        true_output = row['Output']
        
        # Get the predicted equation and replace placeholders with actual numbers
        predicted_equation = replace_numbers_in_equation(predicted_equations[i], input_numbers)
        predicted_output = evaluate_equation(predicted_equation)
        
        # Check if the predicted output matches the true output
        if predicted_output == true_output:
            correct_predictions += 1
        
        total_predictions += 1
    
    return correct_predictions / total_predictions

# Training loop with model saving and accuracy calculation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 100
best_val_acc = 0


# In[13]:


label_smoothing = 0.1

for epoch in tqdm(range(num_epochs)):
    model.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Apply label smoothing
        smoothed_loss = (1 - label_smoothing) * loss + label_smoothing * \
            torch.log(torch.tensor(
                model.config.vocab_size).float()).to(device)

        train_loss += smoothed_loss.item()

        smoothed_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}")

    # Validation step
    model.eval()
    val_loss = 0
    predicted_equations = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            val_loss += model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss.item()

            # Convert generated tokens back to equations
            predicted_equations.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs])

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss}")

    # Save the model if validation loss decreases

    # Calculate accuracy after each epoch using validation data
    val_accuracy = calculate_accuracy(predicted_equations, validation_df)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    
    if best_val_acc < val_accuracy:
        best_val_acc = val_accuracy
        model.save_pretrained('./best_t5_arithmetic_model')
        tokenizer.save_pretrained('./best_t5_arithmetic_model')
        print(f"Model saved at epoch {epoch + 1}")

# Save the final model
model.save_pretrained('./final_t5_arithmetic_model')
tokenizer.save_pretrained('./final_t5_arithmetic_model')


# In[ ]:




