import torch.nn as nn
import torch 

class SimpleFFNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_matrix):
        super(SimpleFFNNClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        hidden = self.relu(self.fc1(pooled))
        output = self.fc2(hidden)
        return output
    
    
class ComplexFFNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dims, output_dim, embedding_matrix, dropout_rate=0.0):
        super(ComplexFFNNClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        
        layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        self.fc_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        features = self.fc_layers(pooled)
        output = self.output_layer(features)
        return output