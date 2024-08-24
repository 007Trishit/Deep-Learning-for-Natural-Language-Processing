import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import requests
import os
import zipfile
from models import *

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the SST2 dataset
sst2 = load_dataset("sst2")

# Download and load GloVe embeddings
def download_glove(url, save_path):
    if not os.path.exists(save_path):
        print("Downloading GloVe embeddings...")
        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")

def load_glove(file_path):
    print("Loading GloVe embeddings...")
    embeddings_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector
    print("GloVe embeddings loaded.")
    return embeddings_dict

glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
glove_path = "glove.6B.zip"
glove_file = "glove.6B.300d.txt"

download_glove(glove_url, glove_path)
with zipfile.ZipFile(glove_path, 'r') as zip_ref:
    zip_ref.extract(glove_file)

glove = load_glove(glove_file)

def tokenize(sentence):
    return sentence.lower().split()

def sentence_to_indices(sentence, vocab, max_length):
    tokens = tokenize(sentence)
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    if len(indices) < max_length:
        indices += [vocab['<pad>']] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    return indices

# Create vocabulary
vocab = {word: idx for idx, word in enumerate(glove.keys())}
vocab['<unk>'] = len(vocab)
vocab['<pad>'] = len(vocab)

embedding_dim = len(next(iter(glove.values())))
embedding_matrix = np.zeros((len(vocab), embedding_dim))
for word, idx in vocab.items():
    if word in glove:
        embedding_matrix[idx] = glove[word]
embedding_matrix = torch.FloatTensor(embedding_matrix)

class SST2Dataset(Dataset):
    def __init__(self, sentences, labels, vocab, max_length):
        self.sentences = sentences
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        indices = sentence_to_indices(sentence, self.vocab, self.max_length)
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# Hyperparameters
max_length = 50
hidden_dims = [512, 256, 128]
output_dim = 2
batch_size = 128
num_epochs = 20
learning_rate = 0.001
dropout_rate = 0.0

# Prepare data
train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    sst2['train']['sentence'], sst2['train']['label'], test_size=0.1, random_state=42
)

train_dataset = SST2Dataset(train_sentences, train_labels, vocab, max_length)
val_dataset = SST2Dataset(val_sentences, val_labels, vocab, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

# Initialize model, loss function, and optimizer
model = ComplexFFNNClassifier(len(vocab), embedding_dim, hidden_dims, output_dim, embedding_matrix, dropout_rate).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

# Training loop
best_accuracy = 0
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch: {epoch+1}, Accuracy: {accuracy:.2f}%')
    
    scheduler.step(accuracy)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_model.pth')

# Load best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate on test set
test_dataset = SST2Dataset(sst2['test']['sentence'], sst2['test']['label'], vocab, max_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')


