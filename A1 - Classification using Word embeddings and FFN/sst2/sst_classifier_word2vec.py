import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api
import pandas as pd
from models import *
from dataset_builders import *
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the SST2 dataset
sst2 = load_dataset("sst2")
print(f"Dataset loaded. Train size: {len(sst2['train'])}, Val Size: {len(sst2['validation'])}")

# Load the custom test dataset
def load_test_data(file_path):
    df = pd.read_csv(file_path, header=None, names=['label', 'text'])
    return df['text'].tolist(), df['label'].tolist()

# Load pre-trained Word2Vec embeddings
print("Loading pre-trained Word2Vec embeddings...")
word2vec_model = api.load("word2vec-google-news-300")
print("Pre-trained Word2Vec embeddings loaded.")

# Preprocessing function
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

def create_vocab_and_embeddings(texts, embedding_model):
    vocab = {'<pad>': 0, '<unk>': 1}
    idx = 2
    for text in texts:
        for token in preprocess_text(text):
            if token not in vocab and token in embedding_model:
                vocab[token] = idx
                idx += 1
    
    embedding_dim = embedding_model.vector_size
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, idx in vocab.items():
        if word in embedding_model:
            embedding_matrix[idx] = embedding_model[word]
        elif word == '<pad>':
            embedding_matrix[idx] = np.zeros(embedding_dim)
        else:  # For <unk> and words not in pre-trained embeddings
            embedding_matrix[idx] = np.random.uniform(-0.25, 0.25, embedding_dim)
    
    return vocab, torch.FloatTensor(embedding_matrix)

def sentence_to_indices(sentence, vocab, max_length):
    tokens = preprocess_text(sentence)
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    if len(indices) < max_length:
        indices += [vocab['<pad>']] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    return indices


embedding_dim = word2vec_model.vector_size
# def create_embedding_matrix(w2v_model, vocab):
#     embedding_matrix = np.zeros((len(vocab), embedding_dim))
#     for word, idx in vocab.items():
#         if word in w2v_model:
#             embedding_matrix[idx] = w2v_model[word]
#         elif word == '<pad>':
#             embedding_matrix[idx] = np.zeros(embedding_dim)
#         else:  # For <unk> and words not in pre-trained embeddings
#             embedding_matrix[idx] = np.random.uniform(-0.25, 0.25, embedding_dim)
#     return torch.FloatTensor(embedding_matrix)

# # Create vocabulary (only include words that are in the pre-trained model)
# vocab = {'<pad>': 0, '<unk>': 1}
# idx = 2
# for sentence in sst2['train']['sentence']:
#     for token in tokenize(sentence):
#         if token not in vocab and token in word2vec_model:
#             vocab[token] = idx
#             idx += 1
# print(f"Vocabulary size: {len(vocab)}")

# # Create embedding matrix
# embedding_matrix = create_embedding_matrix(word2vec_model, vocab)
# print(f"Embedding matrix shape: {embedding_matrix.shape}")

# Hyperparameters
max_length = 50
# hidden_dim = 256
hidden_dims = [1024, 512, 256, 128, 64]
output_dim = 2
batch_size = 64
num_epochs = 20
learning_rate = 0.001



# Prepare data
train_sentences, val_sentences, train_labels, val_labels = sst2['train']['sentence'], sst2['validation']['sentence'], sst2['train']['label'], sst2['validation']['label']

test_sentences, test_labels = load_test_data('SST2_TestData.csv')

# Create vocabulary and embedding matrix
vocab, embedding_matrix = create_vocab_and_embeddings(train_sentences, word2vec_model)
print(f"Vocabulary size: {len(vocab)}")

train_dataset = SST2Dataset(train_sentences, train_labels, vocab, max_length)
val_dataset = SST2Dataset(val_sentences, val_labels, vocab, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

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

# Initialize model, loss function, and optimizer
model = ComplexFFNNClassifier(len(vocab), embedding_dim, hidden_dims, output_dim, embedding_matrix).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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