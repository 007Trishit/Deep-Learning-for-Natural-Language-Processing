import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import re
import pickle

# 1. Data Preparation
def read_data(file_path):
    inputs = []
    targets = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                input_seq, target_seq = line.split(',', 1)
                inputs.append(input_seq.strip()[1:-1])
                targets.append(target_seq.strip()[1:-1])
    return inputs, targets


train_inputs, train_targets = read_data('Assignment2_train.txt')
val_inputs, val_targets = read_data('Assignment2_validation.txt')


def tokenize_input(text):
    tokens = re.findall(r'\d+|[a-zA-Z]+', text.lower())
    return tokens

def tokenize_target(text):
    return list(text)

from collections import Counter

def build_vocab(token_lists):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>':2, '<UNK>':3}
    for token in counter:
        vocab[token] = len(vocab)
    return vocab

input_tokens = [tokenize_input(seq) for seq in train_inputs]
target_tokens = [tokenize_target(seq) for seq in train_targets]

input_vocab = build_vocab(input_tokens)
output_vocab = build_vocab(target_tokens)

input_vocab_size = len(input_vocab)
output_vocab_size = len(output_vocab)

def encode_input(tokens, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

def encode_target(tokens, vocab):
    return [vocab['<SOS>']] + [vocab.get(token, vocab['<UNK>']) for token in tokens] + [vocab['<EOS>']]

def pad_sequences(sequences, max_len):
    return [seq + [0]*(max_len - len(seq)) for seq in sequences]

max_input_len = max(len(seq) for seq in input_tokens)
max_target_len = max(len(seq) for seq in target_tokens) + 2  # For <SOS> and <EOS>

train_input_indices = [encode_input(seq, input_vocab) for seq in input_tokens]
train_target_indices = [encode_target(seq, output_vocab) for seq in target_tokens]

train_input_padded = pad_sequences(train_input_indices, max_input_len)
train_target_padded = pad_sequences(train_target_indices, max_target_len)

val_input_tokens = [tokenize_input(seq) for seq in val_inputs]
val_target_tokens = [tokenize_target(seq) for seq in val_targets]

val_input_indices = [encode_input(seq, input_vocab) for seq in val_input_tokens]
val_target_indices = [encode_target(seq, output_vocab) for seq in val_target_tokens]

val_input_padded = pad_sequences(val_input_indices, max_input_len)
val_target_padded = pad_sequences(val_target_indices, max_target_len)

class DateDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

batch_size = 64

train_dataset = DateDataset(train_input_padded, train_target_padded)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

val_dataset = DateDataset(val_input_padded, val_target_padded)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

# 2. Model Implementation
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden, encoder_outputs):
        hidden_with_time_axis = hidden.unsqueeze(1)
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(hidden_with_time_axis)))
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * encoder_outputs
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector, attention_weights

class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super(LuongAttention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden, encoder_outputs):
        score = torch.bmm(encoder_outputs, hidden.unsqueeze(2)).squeeze(2)
        attention_weights = torch.softmax(score, dim=1).unsqueeze(1)
        context_vector = torch.bmm(attention_weights, encoder_outputs).squeeze(1)
        return context_vector, attention_weights

class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, attention_type='bahdanau'):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.attention_type = attention_type
        if attention_type == 'bahdanau':
            self.attention = BahdanauAttention(hidden_size)
        else:
            self.attention = LuongAttention(hidden_size)
    
    def forward(self, x, hidden, encoder_outputs):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)
        if self.attention_type == 'bahdanau':
            context_vector, attention_weights = self.attention(hidden[-1], encoder_outputs)
        else:
            context_vector, attention_weights = self.attention(hidden[-1], encoder_outputs)
        rnn_input = torch.cat([embedded, context_vector.unsqueeze(1)], dim=-1)
        output, hidden = self.gru(rnn_input, hidden)
        output = output.squeeze(1)
        output = self.fc(output)
        return output, hidden, attention_weights

# 3. Training Procedure
embed_size = 256
hidden_size = 512

attention_type = 'bahdanau'
# attention_type = 'luong'

encoder = Encoder(input_vocab_size, embed_size, hidden_size).to('cuda')
decoder = Decoder(output_vocab_size, embed_size, hidden_size, attention_type=attention_type).to('cuda')

criterion = nn.CrossEntropyLoss(ignore_index=0)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)

num_epochs = 10
teacher_forcing_ratio = 0.5

for epoch in range(1, num_epochs+1):
    encoder.train()
    decoder.train()
    total_loss = 0
    for batch in train_loader:
        input_batch, target_batch = zip(*batch)
        input_batch = pad_sequence(input_batch, batch_first=True).to('cuda')
        target_batch = pad_sequence(target_batch, batch_first=True).to('cuda')
        batch_size = input_batch.size(0)
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        encoder_outputs, encoder_hidden = encoder(input_batch)
        
        decoder_input = torch.tensor([output_vocab['<SOS>']]*batch_size).to('cuda')
        decoder_hidden = encoder_hidden
        
        max_target_len = target_batch.size(1)
        loss = 0
        
        use_teacher_forcing = True if np.random.rand() < teacher_forcing_ratio else False
        
        if use_teacher_forcing:
            for t in range(1, max_target_len):
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_batch[:, t])
                decoder_input = target_batch[:, t]
        else:
            for t in range(1, max_target_len):
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                loss += criterion(decoder_output, target_batch[:, t])
        
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        total_loss += loss.item() / max_target_len
    avg_loss = total_loss / len(train_loader)
    
    # Compute validation loss
    encoder.eval()
    decoder.eval()
    val_loss = 0
    with torch.no_grad():
        for val_batch in val_loader:
            input_batch, target_batch = zip(*val_batch)
            input_batch = pad_sequence(
                input_batch, batch_first=True).to('cuda')
            target_batch = pad_sequence(
                target_batch, batch_first=True).to('cuda')
            batch_size = input_batch.size(0)

            encoder_outputs, encoder_hidden = encoder(input_batch)
            decoder_input = torch.tensor(
                [output_vocab['<SOS>']]*batch_size).to('cuda')
            decoder_hidden = encoder_hidden
            max_target_len = target_batch.size(1)
            loss = 0

            # Use teacher forcing during validation
            for t in range(1, max_target_len):
                decoder_output, decoder_hidden, _ = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_batch[:, t])
                decoder_input = target_batch[:, t]

            val_loss += loss.item() / max_target_len
    avg_val_loss = val_loss / len(val_loader)

    print(f'Epoch {epoch}, Training Loss: {
          avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')


# Save the models
torch.save(encoder.state_dict(), 'encoder_model.pth')
torch.save(decoder.state_dict(), 'decoder_model.pth')

# Save the vocabularies
with open('input_vocab.pkl', 'wb') as f:
    pickle.dump(input_vocab, f)
with open('output_vocab.pkl', 'wb') as f:
    pickle.dump(output_vocab, f)

# 4. Evaluation and Metrics Computation
def evaluate(encoder, decoder, input_seq):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        input_indices = encode_input(tokenize_input(input_seq), input_vocab)
        input_tensor = torch.tensor(input_indices).unsqueeze(0).to('cuda')
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_input = torch.tensor([output_vocab['<SOS>']]).to('cuda')
        decoder_hidden = encoder_hidden
        decoded_tokens = []
        for t in range(10):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            predicted_id = topi.item()
            if predicted_id == output_vocab['<EOS>']:
                break
            decoded_tokens.append(predicted_id)
            decoder_input = topi.squeeze(1).detach()
        predicted_seq = ''.join([list(output_vocab.keys())[list(
            output_vocab.values()).index(idx)] for idx in decoded_tokens])
    return predicted_seq


exact_match_errors = 0
total_mismatches = 0
position_mismatches = [0]*10  # Positions 1 to 10
total_sequences = len(val_inputs)

for idx in range(total_sequences):
    input_seq = val_inputs[idx]
    target_seq = val_targets[idx]
    predicted_seq = evaluate(encoder, decoder, input_seq)

    # Ensure both sequences are exactly 10 characters
    target_seq = target_seq[:10]  # Should already be 10 characters
    predicted_seq = predicted_seq[:10]

    # Pad predicted_seq with a special character if it's shorter than 10
    if len(predicted_seq) < 10:
        # '#' represents missing characters
        predicted_seq += '#' * (10 - len(predicted_seq))

    # Question 1: Exact match error
    if predicted_seq != target_seq:
        exact_match_errors += 1

    # Question 2: Number of mismatches
    mismatches = 0
    for i in range(10):
        if i < len(predicted_seq):
            pred_char = predicted_seq[i]
        else:
            pred_char = '#'  # Placeholder for missing characters
        target_char = target_seq[i]
        if pred_char != target_char:
            mismatches += 1
            position_mismatches[i] += 1
    total_mismatches += mismatches

# for idx in range(total_sequences):
#     input_seq = val_inputs[idx]
#     target_seq = val_targets[idx]
#     predicted_seq = evaluate(encoder, decoder, input_seq)
#     predicted_seq = predicted_seq.ljust(10)[:10]
#     target_seq = target_seq.ljust(10)[:10]
#     if predicted_seq != target_seq:
#         exact_match_errors += 1
#     mismatches = sum(1 for a, b in zip(predicted_seq, target_seq) if a != b)
#     total_mismatches += mismatches
#     for i in range(10):
#         if predicted_seq[i] != target_seq[i]:
#             position_mismatches[i] += 1

# Question 1: Average Validation Set Error (%) using "Exact Match over all 10 outputs"
error_q1 = (exact_match_errors / total_sequences) * 100

# Question 2: Average Validation Set Error (%) (number of mismatches averaged over all 10 outputs)
error_q2 = (total_mismatches / (total_sequences * 10)) * 100

# Questions 3 and 4: Mismatches per position
avg_position_errors = [
    count / total_sequences for count in position_mismatches]
highest_error_position = avg_position_errors.index(
    max(avg_position_errors)) + 1
lowest_error_position = avg_position_errors.index(min(avg_position_errors)) + 1

print(f'Question 1: Average Validation Set Error (%): {error_q1:.2f}%')
print(f'Question 2: Average Validation Set Error (%): {error_q2:.2f}%')
print(f'Question 3: Highest error at output position: {
      highest_error_position}')
print(f'Question 4: Lowest error at output position: {lowest_error_position}')

output_vocab_inv = {index: token for token, index in output_vocab.items()}

def evaluate_with_attention(encoder, decoder, input_seq):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        input_tokens = tokenize_input(input_seq)
        input_indices = encode_input(input_tokens, input_vocab)
        input_tensor = torch.tensor(input_indices).unsqueeze(0).to('cuda')
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_input = torch.tensor([output_vocab['<SOS>']]).to('cuda')
        decoder_hidden = encoder_hidden
        decoded_tokens = []
        attentions = []
        max_output_length = 10  # Adjust as needed
        for t in range(max_output_length):
            decoder_output, decoder_hidden, attention_weights = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # Remove batch dimension
            attentions.append(attention_weights.cpu().numpy().squeeze(0))
            topv, topi = decoder_output.topk(1)
            predicted_id = topi.item()
            if predicted_id == output_vocab['<EOS>']:
                break
            decoded_tokens.append(predicted_id)
            decoder_input = topi.squeeze(1).detach()
        predicted_seq = ''.join([output_vocab_inv[idx]
                                for idx in decoded_tokens])
    return predicted_seq, attentions, input_tokens


def visualize_attention(input_tokens, output_tokens, attentions, input_text):

    # Shape: (output_length, input_length)
    attention_matrix = np.stack(attentions)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(attention_matrix, cmap='viridis')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticks(range(len(input_tokens)))
    ax.set_xticklabels(input_tokens, rotation=90)
    ax.set_yticks(range(len(output_tokens)))
    ax.set_yticklabels(output_tokens)

    # Optionally, adjust tick parameters for better visibility
    ax.tick_params(labelsize=12)

    plt.xlabel('Input Sequence', fontsize=14)
    plt.ylabel('Output Sequence', fontsize=14)
    plt.title('Attention Weights', fontsize=16)
    plt.show()
    plt.savefig(f"{input_text}.png", dpi=300)


# Process and visualize the examples
examples = [
    ("saturday 29 february 2021", "2021-02-29"),
    ("29 february 2020", "2020-02-29")
]

for input_seq, target_seq in examples:
    predicted_seq, attentions, input_tokens = evaluate_with_attention(
        encoder, decoder, input_seq)
    output_tokens = list(predicted_seq)
    print(f"Input Sequence: {input_seq}")
    print(f"Target Sequence: {target_seq}")
    print(f"Predicted Sequence: {predicted_seq}")
    visualize_attention(input_tokens, output_tokens, attentions, input_seq)
