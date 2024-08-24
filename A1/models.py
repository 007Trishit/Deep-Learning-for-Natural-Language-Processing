import torch.nn as nn
import torch 

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
    
    
class ComplexFFNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dims, output_dim, embedding_matrix, dropout_rate=0.5):
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