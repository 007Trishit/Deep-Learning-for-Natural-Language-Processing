from torch.utils.data import Dataset, DataLoader
import torch


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