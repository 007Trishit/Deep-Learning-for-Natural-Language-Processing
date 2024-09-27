import pandas as pd
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import openpyxl

# Load datasets
train_data = pd.read_excel('ArithOps_Train.xlsx')
valid_data = pd.read_excel('ArithOps_Validation.xlsx')
test_data = pd.read_excel('SampleTestFile.xlsx')

# Preprocessing to replace placeholders like number0, number1 with actual input numbers


def preprocess_data(df):
    updated_descriptions = []
    updated_expressions = []

    for _, row in df.iterrows():
        description = row['Description']
        input_numbers = row['Input Numbers'].split(' ')  # Assuming numbers are comma-separated
        expression = row['Equation']

        # Convert expression to string in case it's a float or NaN
        if not isinstance(expression, str):
            expression = str(expression)
            
        # Replace placeholders like number0, number1, etc. with actual input numbers
        for idx, num in enumerate(input_numbers):
            description = description.replace(f'number{idx}', num)
            expression = expression.replace(f'number{idx}', num)

        updated_descriptions.append(description)
        updated_expressions.append(expression)

    updated_df = pd.DataFrame(columns=['Description', 'Question', 'Equation', 'Output'])
    updated_df['Description'] = updated_descriptions
    updated_df['Equation'] = updated_expressions
    updated_df['Output'] = df['Output']
    updated_df['Question'] = df['Question']
    return updated_df


# Preprocess train and validation data
train_data = preprocess_data(train_data)
valid_data = preprocess_data(valid_data)

# Save updated data to new Excel files
train_data.to_excel('Updated_ArithOps_Train.xlsx', index=False)
valid_data.to_excel('Updated_ArithOps_Validation.xlsx', index=False)

# Dataset class for question answering task


class ArithmeticQADataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        question = row['Description']
        answer = row['Equation']

        # Tokenize the input question and answer
        inputs = self.tokenizer.encode_plus(
            question,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True
        )

        # Tokenize the answer (equation) as a string
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            # Modify for specific start positions if needed
            'start_positions': torch.tensor(0, dtype=torch.long),
            'end_positions': torch.tensor(len(answer_ids)-1, dtype=torch.long)
        }


# Initialize tokenizer and model (use a BERT-based model for question answering)
tokenizer = BertTokenizer.from_pretrained(
    'deepset/bert-large-uncased-whole-word-masking-squad2')
model = BertForQuestionAnswering.from_pretrained(
    'deepset/bert-large-uncased-whole-word-masking-squad2')

# Create datasets and dataloaders
train_dataset = ArithmeticQADataset(train_data, tokenizer)
valid_dataset = ArithmeticQADataset(valid_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)

# Setup optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)


# Train the model
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f"Train loss: {train_loss}")

# Prediction on validation data


def eval_model(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            # Get the predicted start and end positions
            start_pred = torch.argmax(start_logits, dim=1).cpu().numpy()
            end_pred = torch.argmax(end_logits, dim=1).cpu().numpy()

            # Decode the predicted equations (postfix or infix)
            for start, end, ids in zip(start_pred, end_pred, input_ids):
                tokens = tokenizer.convert_ids_to_tokens(ids[start:end+1])
                equation = tokenizer.convert_tokens_to_string(tokens)
                predictions.append(equation)

    return predictions


# Get validation predictions
val_preds = eval_model(model, valid_loader, device)

# Evaluate the equations (you can further implement postfix/infix evaluation here)

# Save predictions to Excel
valid_data['Predicted Equation'] = val_preds
valid_data.to_excel('Predicted_Validation_Results.xlsx', index=False)

# Test data preprocessing and prediction
test_data = preprocess_data(test_data)


def predict_on_test(test_data, model, tokenizer, device):
    model.eval()
    predictions = []

    for _, row in test_data.iterrows():
        description = row['Description']

        inputs = tokenizer.encode_plus(
            description,
            None,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True
        )

        input_ids = torch.tensor(
            [inputs['input_ids']], dtype=torch.long).to(device)
        attention_mask = torch.tensor(
            [inputs['attention_mask']], dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            start_pred = torch.argmax(start_logits, dim=1).cpu().numpy()[0]
            end_pred = torch.argmax(end_logits, dim=1).cpu().numpy()[0]

            tokens = tokenizer.convert_ids_to_tokens(
                input_ids[0][start_pred:end_pred+1])
            equation = tokenizer.convert_tokens_to_string(tokens)
            predictions.append(equation)

    # Save predictions to Excel
    test_data['Predicted Equation'] = predictions
    test_data.to_excel('Predicted_Test_Results.xlsx', index=False)


# Predict on test data
predict_on_test(test_data, model, tokenizer, device)
