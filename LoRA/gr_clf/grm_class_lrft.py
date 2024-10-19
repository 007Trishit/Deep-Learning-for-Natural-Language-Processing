import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Define the dataset class


class SentenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = self.data.iloc[index]['document_content']
        label = self.data.iloc[index]['label']

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Load and preprocess data
train_df = pd.read_csv('data/in_domain_train.csv')
val_df = pd.read_csv('data/in_domain_dev.csv')
test_df = pd.read_csv('data/out_of_domain_dev.csv')

# Function to train and evaluate model

RANK = 8
targets = "qvk"

def train_and_evaluate(model_name, train_df, val_df, test_df):
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2)

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=RANK,
        target_modules=["query_proj", "value_proj", "key_proj"],
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    # Prepare datasets
    train_dataset = SentenceDataset(train_df, tokenizer, max_length=128)
    val_dataset = SentenceDataset(val_df, tokenizer, max_length=128)
    test_dataset = SentenceDataset(test_df, tokenizer, max_length=128)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./saved_models/{model_name}_{RANK}_{targets}", 
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"./logs/{model_name}",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        load_best_model_at_end=True,
    )

    # Define compute_metrics function
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)

    return test_results


# List of models to fine-tune
models = [
    "roberta-base",
    "microsoft/deberta-base",
    "facebook/bart-base"
]

# Fine-tune and evaluate each model
for model_name in models:
    print(f"Fine-tuning {model_name}...")
    test_results = train_and_evaluate(model_name, train_df, val_df, test_df)
    print(f"Test results for {model_name}:")
    print(test_results)
    print("\n")
