import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
from models import *
from dataset_builders import *

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the SST2 dataset
sst2 = load_dataset("stanfordnlp/sst2")
print(f"Dataset loaded. Train size: {len(sst2['train'])+len(sst2['validation'])}, Val and Test size: {0.1*(len(sst2['train'])+len(sst2['validation']))}")

# Download and load GloVe embeddings (simplified for brevity)
def load_glove(file_path):
    print("Loading GloVe embeddings...")
    embeddings_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector
            if i % 100000 == 0:
                print(f"Processed {i} words...")
    print(f"GloVe embeddings loaded. Total words: {len(embeddings_dict)}")
    return embeddings_dict

# For debugging, we'll use a smaller subset of GloVe
glove = load_glove('glove.6B.300d.txt')  # Make sure this file exists in your directory

# Create vocabulary
vocab = {word: idx for idx, word in enumerate(glove.keys())}
vocab['<unk>'] = len(vocab)
vocab['<pad>'] = len(vocab)
print(f"Vocabulary size: {len(vocab)}")

embedding_dim = len(next(iter(glove.values())))
embedding_matrix = np.zeros((len(vocab), embedding_dim))
for word, idx in vocab.items():
    if word in glove:
        embedding_matrix[idx] = glove[word]
embedding_matrix = torch.FloatTensor(embedding_matrix)
print(f"Embedding matrix shape: {embedding_matrix.shape}")

# Hyperparameters
max_length = 50
# hidden_dim = 256
hidden_dims = [1024, 512, 256, 128, 64]
output_dim = 2
batch_size = 128
num_epochs = 20
learning_rate = 0.001

# Prepare data
train_sentences, eval_sentences, train_labels, eval_labels = train_test_split(
    sst2['train']['sentence']+sst2['validation']['sentence'], sst2['train']['label']+sst2['validation']['label'], test_size=0.2, random_state=42
)

test_sentences, val_sentences, test_labels, val_labels = train_test_split(
    eval_sentences, eval_labels, test_size=0.5, random_state=42
)

train_dataset = SST2Dataset(train_sentences, train_labels, vocab, max_length)
val_dataset = SST2Dataset(val_sentences, val_labels, vocab, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize model, loss function, and optimizer
model = ComplexFFNNClassifier(len(vocab), embedding_dim, hidden_dims, output_dim, embedding_matrix).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    train_accuracy = evaluate(model, train_loader)
    val_accuracy = evaluate(model, val_loader)
    
    print(f'Epoch {epoch+1}, Average Loss: {total_loss/len(train_loader):.4f}')
    print(f'Training Accuracy: {train_accuracy:.2f}%')
    print(f'Validation Accuracy: {val_accuracy:.2f}%')
    print('-' * 50)

# Evaluate on test set
test_dataset = SST2Dataset(test_sentences, test_labels, vocab, max_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

test_accuracy = evaluate(model, test_loader)
print(f'Test Accuracy: {test_accuracy:.2f}%')

# Debugging: Check predictions
print("Debugging: Checking predictions")
model.eval()
with torch.no_grad():
    for i, (data, target) in enumerate(test_loader):
        if i >= 5:  # Check first 5 batches
            break
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        print(f"Batch {i+1}:")
        print(f"Predictions: {predicted[:10]}")
        print(f"Actual labels: {target[:10]}")
        print(f"Correct: {(predicted == target).sum().item()}/{len(target)}")
        print()

# Check vocabulary coverage
print("Checking vocabulary coverage")
vocab_hits = 0
vocab_misses = 0
for sentence in sst2['test']['sentence'][:1000]:  # Check first 1000 sentences
    for word in tokenize(sentence):
        if word in vocab:
            vocab_hits += 1
        else:
            vocab_misses += 1
print(f"Vocabulary hits: {vocab_hits}")
print(f"Vocabulary misses: {vocab_misses}")
print(f"Vocabulary coverage: {vocab_hits / (vocab_hits + vocab_misses) * 100:.2f}%")