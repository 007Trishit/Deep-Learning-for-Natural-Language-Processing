import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api
from collections import Counter

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the custom dataset
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df['reviewText'].astype(str).tolist(), df['overall'].tolist()

# Load pre-trained word embeddings
def load_embeddings(embedding_type='word2vec'):
    print(f"Loading pre-trained {embedding_type} embeddings...")
    if embedding_type == 'word2vec':
        model = api.load("word2vec-google-news-300")
    elif embedding_type == 'glove':
        model = api.load("glove-wiki-gigaword-300")
    elif embedding_type == 'fasttext':
        model = api.load("fasttext-wiki-news-subwords-300")
    else:
        raise ValueError("Invalid embedding type. Choose 'word2vec', 'glove', or 'fasttext'.")
    print("Embeddings loaded.")
    return model

# Tokenization function
def tokenize(text):
    # You might want to improve this tokenization based on your specific needs
    return text.lower().split()

# Create vocabulary and embedding matrix
def create_vocab_and_embeddings(texts, embedding_model):
    vocab = {'<pad>': 0, '<unk>': 1}
    idx = 2
    for text in texts:
        for token in tokenize(text):
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

# Convert text to indices
def text_to_indices(text, vocab, max_length):
    tokens = tokenize(text)
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    if len(indices) < max_length:
        indices += [vocab['<pad>']] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    return indices

# Custom Dataset
class CustomTextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        indices = text_to_indices(text, self.vocab, self.max_length)
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# FFN Model
class FFNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_matrix):
        super(FFNClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        hidden1 = self.dropout(self.relu(self.fc1(pooled)))
        hidden2 = self.dropout(self.relu(self.fc2(hidden1)))
        output = self.fc3(hidden2)
        return output


# Training function
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_accuracy = 0
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
        # Validation
        val_accuracy = evaluate(model, val_loader)
        print(f'Epoch {epoch+1}, Average Loss: {total_loss/len(train_loader):.4f}')
        print(f'Training Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        print('-' * 50)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f'models/ov_samp/best_model_{embedding_type}.pth')
        

# Evaluation function
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

# Main execution
if __name__ == "__main__":
    # Hyperparameters
    max_length = 100
    hidden_dim = 256
    batch_size = 64
    num_epochs = 30
    learning_rate = 0.001
    embedding_type = 'fasttext'  # Change this to 'glove' or 'fasttext' if needed

    # Load train data
    train_texts, train_labels = load_data('Aug24-Assignmen1-Dataset1.xlsx')

    # Handle class imbalance
    print("Class distribution before oversampling:")
    print(Counter(train_labels))
    ros = RandomOverSampler(random_state=42)
    train_texts_resampled, train_labels_resampled = ros.fit_resample(
        pd.DataFrame(train_texts), train_labels
    )
    train_texts_resampled = train_texts_resampled.iloc[:, 0].tolist()
    print("Class distribution after oversampling:")
    print(Counter(train_labels_resampled))

    # Load embeddings and create vocabulary
    embedding_model = load_embeddings(embedding_type)
    vocab, embedding_matrix = create_vocab_and_embeddings(train_texts_resampled, embedding_model)
    print(f"Vocabulary size: {len(vocab)}")

    # Encode labels
    le = LabelEncoder()
    train_labels_encoded = le.fit_transform(train_labels_resampled)
    num_classes = len(le.classes_)

    # Create datasets and dataloaders
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_texts_resampled, train_labels_encoded, test_size=0.1, random_state=42, stratify=train_labels_encoded
    )

    train_dataset = CustomTextDataset(train_data, train_labels, vocab, max_length)
    val_dataset = CustomTextDataset(val_data, val_labels, vocab, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = FFNClassifier(len(vocab), embedding_model.vector_size, hidden_dim, num_classes, embedding_matrix).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs)

    # Load test dat, best model and evaluate on test set
    test_texts, test_labels = load_data('Aug24-Assignmen1-Dataset1-test.xlsx')
    test_labels_encoded = le.transform(test_labels)
    test_dataset = CustomTextDataset(test_texts, test_labels_encoded, vocab, max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model.load_state_dict(torch.load(f'models/ov_samp/best_model_{embedding_type}.pth'))
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
            print(f"Predictions: {le.inverse_transform(predicted.cpu().numpy())[:10]}")
            print(f"Actual labels: {le.inverse_transform(target.cpu().numpy())[:10]}")
            print(f"Correct: {(predicted == target).sum().item()}/{len(target)}")
            print()

    # Check vocabulary coverage
    print("Checking vocabulary coverage")
    vocab_hits = 0
    vocab_misses = 0
    for text in test_texts[:1000]:  # Check first 1000 texts
        for word in tokenize(text):
            if word in vocab:
                vocab_hits += 1
            else:
                vocab_misses += 1
    print(f"Vocabulary hits: {vocab_hits}")
    print(f"Vocabulary misses: {vocab_misses}")
    print(f"Vocabulary coverage: {vocab_hits / (vocab_hits + vocab_misses) * 100:.2f}%")