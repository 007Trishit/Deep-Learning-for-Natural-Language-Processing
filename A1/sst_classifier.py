from datasets import load_dataset
import gensim.downloader as api
import spacy
import torch
import torch.nn as nn
import torch.optim as optim

# Load SST-2 dataset
dataset = load_dataset('sst2')

# Split the dataset into train and test
train_data = dataset['train']
test_data = dataset['validation']


# Load pre-trained word2vec embeddings
word_vectors = api.load("word2vec-google-news-300")  # 300-dimensional vectors

# Load spaCy model for English
nlp = spacy.load("en_core_web_sm")

# Tokenize the sentences
def tokenize(text):
    return [token.text for token in nlp(text)]

# Example of tokenizing and converting to embeddings
def get_embedding(sentence, model):
    tokens = tokenize(sentence)
    vectors = [model[token] for token in tokens if token in model]
    return vectors

# Apply embedding to dataset
def preprocess_dataset(data, model):
    embeddings = []
    labels = []
    for example in data:
        emb = get_embedding(example['sentence'], model)
        if len(emb) > 0:
            embeddings.append(torch.tensor(emb).mean(dim=0))  # Average embedding
            labels.append(example['label'])
    return torch.stack(embeddings), torch.tensor(labels)

train_embeddings, train_labels = preprocess_dataset(train_data, word_vectors)
test_embeddings, test_labels = preprocess_dataset(test_data, word_vectors)

class FFNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

# Model parameters
input_dim = 300  # Dimension of word embeddings
hidden_dim = 128
output_dim = 1  # Binary classification
device = "cuda:0"

model = FFNNClassifier(input_dim, hidden_dim, output_dim).to(device)

criterion = nn.BCELoss()
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
    
    
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Apply t-SNE
tsne = TSNE(n_components=2)
train_embeddings_2d = tsne.fit_transform(train_embeddings)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(train_embeddings_2d[:, 0], train_embeddings_2d[:, 1], c=train_labels, cmap='coolwarm', alpha=0.7)
plt.colorbar()
plt.title("t-SNE Visualization of Sentence Embeddings")
plt.show()