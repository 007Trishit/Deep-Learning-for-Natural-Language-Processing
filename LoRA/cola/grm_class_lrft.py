import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BartForSequenceClassification, GPT2ForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse


# Set the CUDA_VISIBLE_DEVICES environment variable to restrict to GPUs 0-6
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"

# Define the dataset class

class SentenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = self.data.iloc[index, 3]
        label = self.data.iloc[index, 1]

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


def train_and_evaluate(model_name, train_df, val_df, test_df, rank, target_modules, target_codes):
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "gpt2" in model_name:
        model = GPT2ForSequenceClassification.from_pretrained(
            model_name, num_labels=2)
        # Add padding token for GPT2
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    elif "bart" in model_name:
        model = BartForSequenceClassification.from_pretrained(model_name, num_labels=2)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.1,
        bias="none",
        target_modules=target_modules
    )
    model = get_peft_model(model, peft_config)

    # Prepare datasets
    train_dataset = SentenceDataset(train_df, tokenizer, max_length=128)
    val_dataset = SentenceDataset(val_df, tokenizer, max_length=128)
    test_dataset = SentenceDataset(test_df, tokenizer, max_length=128)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./saved_models/{model_name}_{rank}_{target_codes}",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"./logs/{model_name}_{rank}_{target_codes}",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        load_best_model_at_end=True,
    )

    # Define compute_metrics function
    def compute_metrics(pred):
        labels = pred.label_ids
        if isinstance(pred.predictions, tuple):
            # For BART, take the first element of the tuple
            preds = pred.predictions[0].argmax(-1)
        else:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune models with LoRA")
    parser.add_argument("--model", type=str, choices=[
                        "roberta", "deberta", "bart", "gpt2"], required=True, help="Model to fine-tune")
    parser.add_argument("--rank", type=int, default=8, help="Rank for LoRA")
    parser.add_argument("--target_modules", type=str, required=True,
                        help="Target modules for LoRA")
    parser.add_argument("--target_codes", type=str, required=True,
                        help="Target codes for LoRA")
    args = parser.parse_args()

    model_map = {
        "roberta": "roberta-base",
        "deberta": "microsoft/deberta-v3-base",
        "bart": "facebook/bart-base",
        "gpt2": "gpt2"
    }

    model_name = model_map[args.model]
    target_modules = args.target_modules.split(" ")

    print(f"Fine-tuning {model_name}...")
    test_results = train_and_evaluate(
        model_name, train_df, val_df, test_df, args.rank, target_modules, args.target_codes)
    print(f"Test results for {model_name}:")
    print(test_results)
