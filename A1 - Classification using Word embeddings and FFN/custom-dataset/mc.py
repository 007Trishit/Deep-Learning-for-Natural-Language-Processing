import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Load GloVe embeddings
def load_glove_embeddings(path):
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

glove_embeddings = load_glove_embeddings('glove.6B.300d.txt')
embedding_dim = 300

# Custom dataset with random oversampling
class TextDataset(Dataset):
    def __init__(self, dataframe, text_col, label_col=None, max_length=100, oversample=False):
        self.max_length = max_length
        
        if label_col:
            self.label_encoder = LabelEncoder()
            labels = self.label_encoder.fit_transform(dataframe[label_col])
            
            if oversample:
                ros = RandomOverSampler(random_state=42)
                resampled_texts, resampled_labels = ros.fit_resample(
                    dataframe[text_col].values.reshape(-1, 1), labels
                )
                self.texts = resampled_texts.flatten()
                self.labels = resampled_labels
            else:
                self.texts = dataframe[text_col].values
                self.labels = labels
        else:
            self.texts = dataframe[text_col].values
            self.labels = None
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        if not isinstance(text, str):
            text = str(text)
        tokens = word_tokenize(text.lower())[:self.max_length]
        
        # Convert tokens to embeddings
        embeddings = [glove_embeddings.get(token, np.zeros(embedding_dim)) for token in tokens]
        embeddings = embeddings + [np.zeros(embedding_dim)] * (self.max_length - len(embeddings))
        embeddings = np.array(embeddings)
        
        if self.labels is not None:
            return torch.FloatTensor(embeddings), torch.LongTensor([self.labels[idx]])
        else:
            return torch.FloatTensor(embeddings)

# Neural Network model (unchanged)
class DeepFFNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5):
        super(DeepFFNN, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.model(x)

# Load data
train_df = pd.read_excel('Aug24-Assignment1-Train-Dataset1.xlsx')
val_df = pd.read_excel('Aug24-Assignment1-Validation-Dataset1.xlsx')
test_df = pd.read_excel('Aug24-Assignment1-Dataset1-test.xlsx', header=None, names=['reviewText'])

# Create datasets and dataloaders
train_dataset = TextDataset(train_df, 'reviewText', 'overall', oversample=True)
val_dataset = TextDataset(val_df, 'reviewText', 'overall')
test_dataset = TextDataset(test_df, 'reviewText')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize model, loss function, and optimizer
input_dim = embedding_dim * 100  # max_length * embedding_dim
hidden_dims = [256, 128, 64]  # Multiple hidden layers
output_dim = len(train_dataset.label_encoder.classes_)

model = DeepFFNN(input_dim, hidden_dims, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, loader, criterion, optimizer):
    model.train()
    for batch in loader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()

# Evaluation function
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return f1_score(all_labels, all_preds, average='micro')

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer)
    val_f1 = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Micro-F1: {val_f1:.4f}")

# Evaluate on test set
model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        outputs = model(batch)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())

# Convert predictions to original labels
test_preds = train_dataset.label_encoder.inverse_transform(test_preds)

# Save test predictions to Excel
test_df['predicted_label'] = test_preds
test_df.to_excel('test_predictions.xlsx', index=False, header=False)

print("Test predictions saved to 'test_predictions.xlsx'")