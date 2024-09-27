import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from tqdm import tqdm

# Function to evaluate prefix expressions


def evaluate_prefix_expression(expr):
    tokens = expr.strip().split()
    stack = []
    for token in reversed(tokens):
        if token.isdigit():
            stack.append(float(token))
        elif token.replace('.', '', 1).isdigit():
            stack.append(float(token))
        else:
            if len(stack) < 2:
                raise ValueError('Invalid expression')
            a = stack.pop()
            b = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                stack.append(a / b)
            else:
                raise ValueError(f'Unknown operator {token}')
    if len(stack) != 1:
        raise ValueError('Invalid expression')
    return stack[0]


# 1. Load data from .xlsx files
train_df = pd.read_excel('Updated_Train_Data.xlsx')
valid_df = pd.read_excel('Updated_Validation_Data.xlsx')

# 2. Preprocess data


def prepare_input_output(df):
    inputs = []
    outputs = []
    for idx, row in df.iterrows():
        input_text = str(row['Description']) + ' ' + str(row['Question'])
        output_text = str(row['Equation'])
        inputs.append(input_text)
        outputs.append(output_text)
    return inputs, outputs


train_inputs, train_outputs = prepare_input_output(train_df)
valid_inputs, valid_outputs = prepare_input_output(valid_df)
valid_expected_outputs = valid_df['Output'].astype(str).tolist()

# 3. Create Dataset


class ArithmeticDataset(Dataset):
    def __init__(self, inputs, outputs, tokenizer, max_length=512):
        self.inputs = inputs
        self.outputs = outputs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        output_text = self.outputs[idx]
        input_encoding = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        output_encoding = self.tokenizer(
            output_text,
            padding='max_length',
            truncation=True,
            max_length=50,
            return_tensors='pt',
        )
        labels = output_encoding['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = - \
            100  # Ignore padding tokens in the loss
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': labels.flatten(),
        }


# Initialize tokenizer and model
model_name = 'google/flan-t5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Create datasets and dataloaders
train_dataset = ArithmeticDataset(train_inputs, train_outputs, tokenizer)
valid_dataset = ArithmeticDataset(valid_inputs, valid_outputs, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 4. Define optimizer
optimizer = AdamW(model.parameters(), lr=1e-4)

# 5. Training loop
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    # Training
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}'):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc='Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(valid_loader)

    print(f'Epoch {
          epoch+1}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

# 6. Evaluation
model.eval()
correct = 0
total = 0
predicted_equations = []
with torch.no_grad():
    for batch in tqdm(valid_loader, desc='Evaluating'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=50,
            num_beams=5,
            early_stopping=True
        )
        decoded_outputs = [tokenizer.decode(
            ids, skip_special_tokens=True) for ids in outputs]
        predicted_equations.extend(decoded_outputs)

# Compare predictions with expected outputs
for pred_eq, expected_output in zip(predicted_equations, valid_expected_outputs):
    try:
        result = evaluate_prefix_expression(pred_eq)
        if result == expected_output:
            correct += 1
    except Exception as e:
        pass
    total += 1
print(f'Validation Accuracy: {100 * correct / total}%')
