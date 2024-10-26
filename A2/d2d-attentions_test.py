import torch
import torch.nn as nn
import numpy as np
import re
import pickle
import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the saved vocabularies
with open('input_vocab.pkl', 'rb') as f:
    input_vocab = pickle.load(f)
with open('output_vocab.pkl', 'rb') as f:
    output_vocab = pickle.load(f)

input_vocab_size = len(input_vocab)
output_vocab_size = len(output_vocab)
output_vocab_inv = {index: token for token, index in output_vocab.items()}


def tokenize_input(text):
    tokens = re.findall(r'\d+|[a-zA-Z]+', text.lower())
    return tokens


def encode_input(tokens, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

# Define the model classes (Encoder, Decoder, Attention) as before


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
        score = self.V(torch.tanh(self.W1(encoder_outputs) +
                       self.W2(hidden_with_time_axis)))
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * encoder_outputs
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector, attention_weights


class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(embed_size + hidden_size,
                          hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.attention = BahdanauAttention(hidden_size)

    def forward(self, x, hidden, encoder_outputs):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)
        context_vector, attention_weights = self.attention(
            hidden[-1], encoder_outputs)
        rnn_input = torch.cat([embedded, context_vector.unsqueeze(1)], dim=-1)
        output, hidden = self.gru(rnn_input, hidden)
        output = output.squeeze(1)
        output = self.fc(output)
        return output, hidden, attention_weights


embed_size = 256
hidden_size = 512

encoder = Encoder(input_vocab_size, embed_size, hidden_size).to(device)
decoder = Decoder(output_vocab_size, embed_size, hidden_size).to(device)

# Load the saved models
encoder.load_state_dict(torch.load('encoder_model.pth', map_location=device))
decoder.load_state_dict(torch.load('decoder_model.pth', map_location=device))

encoder.eval()
decoder.eval()


def evaluate_test(encoder, decoder, input_seq):
    with torch.no_grad():
        input_tokens = tokenize_input(input_seq)
        input_indices = encode_input(input_tokens, input_vocab)
        input_tensor = torch.tensor(input_indices).unsqueeze(0).to(device)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_input = torch.tensor([output_vocab['<SOS>']], device=device)
        decoder_hidden = encoder_hidden
        decoded_tokens = []
        max_output_length = 10  # Ensure maximum length is 10
        for _ in range(max_output_length):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            predicted_id = topi.item()
            if predicted_id == output_vocab['<EOS>']:
                break
            decoded_tokens.append(predicted_id)
            decoder_input = topi.squeeze(1).detach()
        predicted_seq = ''.join([output_vocab_inv.get(idx, '')
                                for idx in decoded_tokens])
    return predicted_seq

# Read the test inputs


# def read_test_inputs(file_path):
#     with open(file_path, 'r') as f:
#         test_inputs = [line.strip()[1:-1] for line in f if line.strip()]
#     return test_inputs


# test_inputs = read_test_inputs('Assignment2_Test.txt')

# # Generate predictions
# predicted_outputs = []

# for input_seq in test_inputs:
#     predicted_seq = evaluate_test(encoder, decoder, input_seq)
#     predicted_outputs.append(predicted_seq)

# # Save predicted outputs to a file
# with open('test_predictions.txt', 'w') as f:
#     for seq in predicted_outputs:
#         f.write(f"'{seq}'" + '\n')

# predicted_outputs = []

# for input_seq in test_inputs:
#     predicted_seq = evaluate_test(encoder, decoder, input_seq)
#     predicted_outputs.append(predicted_seq)

# # Save predicted outputs to an Excel file
# df_predictions = pd.DataFrame(predicted_outputs)
# # Write to Excel file
# df_predictions.to_excel('test_predictions_1.xlsx', header=False, index=False)

# print("Predictions have been saved to 'test_predictions.xlsx'")


# def read_test_labels(file_path):
#     with open(file_path, 'r') as f:
#         test_labels = [line.strip() for line in f if line.strip()]
#     return test_labels

# test_labels = read_test_labels('test_labels.txt')

# # Compute metrics
# exact_match_errors = 0
# total_mismatches = 0
# position_mismatches = [0]*10
# total_sequences = len(test_labels)

# assert len(predicted_outputs) == total_sequences, "Mismatch between number of predictions and labels"

# for idx in range(total_sequences):
#     target_seq = test_labels[idx]
#     predicted_seq = predicted_outputs[idx]
    
#     target_seq = target_seq[:10]
#     predicted_seq = predicted_seq[:10]
#     if len(predicted_seq) < 10:
#         predicted_seq += '#' * (10 - len(predicted_seq))
    
#     if predicted_seq != target_seq:
#         exact_match_errors += 1
    
#     mismatches = 0
#     for i in range(10):
#         pred_char = predicted_seq[i]
#         target_char = target_seq[i]
#         if pred_char != target_char:
#             mismatches += 1
#             position_mismatches[i] += 1
#     total_mismatches += mismatches

# error_q1 = (exact_match_errors / total_sequences) * 100
# error_q2 = (total_mismatches / (total_sequences * 10)) * 100

# avg_position_errors = [count / total_sequences for count in position_mismatches]
# highest_error_position = avg_position_errors.index(max(avg_position_errors)) + 1
# lowest_error_position = avg_position_errors.index(min(avg_position_errors)) + 1

# print(f'Question 1: Average Test Set Error (%): {error_q1:.2f}%')
# print(f'Question 2: Average Test Set Error (%): {error_q2:.2f}%')
# print(f'Question 3: Highest error at output position: {highest_error_position}')
# print(f'Question 4: Lowest error at output position: {lowest_error_position}')

# Read test labels from the Excel file
def read_test_labels_excel(file_path):
    df_labels = pd.read_excel(file_path, header=None)
    test_labels = df_labels[0].astype(
        str).tolist()  # Ensure labels are strings
    return test_labels


# Read the test inputs and labels from a single text file
def read_test_data(file_path):
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


test_inputs, test_labels = read_test_data('Assignment2_LabeledTestSet.txt')

# Generate predictions
predicted_outputs = []

for input_seq in test_inputs:
    predicted_seq = evaluate_test(encoder, decoder, input_seq)
    predicted_outputs.append(predicted_seq)

# Save predicted outputs to a text file
with open('test_predictions.txt', 'w') as f:
    for seq in predicted_outputs:
        f.write(seq + '\n')

print("Predictions have been saved to 'test_predictions.txt'.")

# Compute metrics
exact_match_errors = 0
total_mismatches = 0
position_mismatches = [0]*10
total_sequences = len(test_labels)

assert len(
    predicted_outputs) == total_sequences, "Mismatch between number of predictions and labels"

for idx in range(total_sequences):
    target_seq = test_labels[idx]
    predicted_seq = predicted_outputs[idx]

    target_seq = target_seq[:10]
    predicted_seq = predicted_seq[:10]
    if len(predicted_seq) < 10:
        predicted_seq += '#' * (10 - len(predicted_seq))
    if len(target_seq) < 10:
        target_seq += '#' * (10 - len(target_seq))

    if predicted_seq != target_seq:
        exact_match_errors += 1

    mismatches = 0
    for i in range(10):
        pred_char = predicted_seq[i]
        target_char = target_seq[i]
        if pred_char != target_char:
            mismatches += 1
            position_mismatches[i] += 1
    total_mismatches += mismatches

error_q1 = (exact_match_errors / total_sequences) * 100
error_q2 = (total_mismatches / (total_sequences * 10)) * 100

avg_position_errors = [
    count / total_sequences for count in position_mismatches]
highest_error_position = avg_position_errors.index(
    max(avg_position_errors)) + 1
lowest_error_position = avg_position_errors.index(min(avg_position_errors)) + 1

print(f'Question 1: Average Test Set Error (%): {error_q1:.2f}%')
print(f'Question 2: Average Test Set Error (%): {error_q2:.2f}%')
print(f'Question 3: Highest error at output position: {
      highest_error_position}')
print(f'Question 4: Lowest error at output position: {lowest_error_position}')
