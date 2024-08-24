from datasets import load_dataset
import gensim.downloader as api
import spacy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models import *

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the SST-2 dataset
dataset = load_dataset('sst2')
train_data = dataset['train']
test_data = dataset['validation']

# Load the pre-trained Word2Vec embeddings using Gensim
word_vectors = api.load("word2vec-google-news-300")
    
# Load spaCy model for English
nlp = spacy.load("en_core_web_sm")

# Tokenize the sentences
def tokenize(text):
    return [token.text for token in nlp(text)]

# Get the Word2Vec embedding for a sentence and move it to GPU
def get_embedding(sentence, model):
    tokens = tokenize(sentence)
    vectors = [torch.tensor(model[token], device=device) for token in tokens if token in model]
    if len(vectors) > 0:
        return torch.mean(torch.stack(vectors), dim=0)  # Average embedding
    else:
        return torch.zeros(300, device=device)  # Return a zero vector if no tokens are found

# Apply embedding to the dataset
def preprocess_dataset(data, model):
    embeddings = []
    labels = []
    for example in data:
        emb = get_embedding(example['sentence'], model)
        embeddings.append(emb)
        labels.append(example['label'])
    return torch.stack(embeddings), torch.tensor(labels, device=device)

train_embeddings, train_labels = preprocess_dataset(train_data, word_vectors)
test_embeddings, test_labels = preprocess_dataset(test_data, word_vectors)

# Model parameters
input_dim = 300  # Dimension of word embeddings
hidden_dim = 128
output_dim = 1  # Binary classification

# Instantiate the model and move it to the GPU
model = FFNNClassifier(input_dim, hidden_dim, output_dim).to(device)

criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(train_embeddings).squeeze()
    loss = criterion(outputs, train_labels.float())
    
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


model.eval()
with torch.no_grad():
    test_outputs = model(test_embeddings).squeeze()
    predicted = (test_outputs > 0.5).float()
    accuracy = (predicted == test_labels).sum() / test_labels.size(0)
    print(f'Test Accuracy: {accuracy.item():.4f}')

# Apply t-SNE
tsne = TSNE(n_components=2)
train_embeddings_2d = tsne.fit_transform(train_embeddings)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(train_embeddings_2d[:, 0], train_embeddings_2d[:, 1], c=train_labels, cmap='coolwarm', alpha=0.7)
plt.colorbar()
plt.title("t-SNE Visualization of Sentence Embeddings")
plt.show()
plt.savefig("t-SNE.png")