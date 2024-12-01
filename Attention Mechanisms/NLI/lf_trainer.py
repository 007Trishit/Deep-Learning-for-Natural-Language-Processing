import torch
from datasets import load_dataset
from transformers import (
    LongformerTokenizerFast,
    LongformerForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
import numpy as np
import logging
import sys
from typing import Dict, List
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


class LongformerMNLITrainer:
    def __init__(
        self,
        model_name: str = "allenai/longformer-base-4096",
        max_length: int = 512,  # Standard length is enough for MNLI
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        output_dir: str = "./longformer_mnli_models",
        attention_window: int = 512
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.output_dir = output_dir
        self.attention_window = attention_window

        # Initialize tokenizer
        self.tokenizer = LongformerTokenizerFast.from_pretrained(model_name)

        # Initialize model with classification head
        self.model = LongformerForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,  # MNLI has 3 classes: entailment, neutral, contradiction
        )

        # Label mapping
        self.label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def preprocess_function(self, examples):
        """Prepare MNLI examples for Longformer"""
        # Tokenize premises and hypotheses
        tokenized = self.tokenizer(
            examples['premise'],
            examples['hypothesis'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )

        # Create global attention mask
        global_attention_mask = torch.zeros(
            (len(tokenized["input_ids"]), self.max_length))

        # Set global attention on [CLS] token and separator tokens
        for idx, input_ids in enumerate(tokenized["input_ids"]):
            # Set global attention on [CLS] token
            global_attention_mask[idx][0] = 1

            # Set global attention on separator tokens
            sep_tokens = (
                torch.tensor(input_ids) == self.tokenizer.sep_token_id
            )
            global_attention_mask[idx][sep_tokens] = 1

        tokenized["global_attention_mask"] = global_attention_mask.tolist()

        # Add labels
        tokenized["labels"] = [
            self.label2id[label] for label in examples['label']
        ]

        return tokenized

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict:
        """Compute metrics for evaluation"""
        predictions = np.argmax(eval_pred.predictions, axis=1)

        return {
            'accuracy': accuracy_score(eval_pred.label_ids, predictions),
            'f1_macro': f1_score(eval_pred.label_ids, predictions, average='macro'),
            'f1_weighted': f1_score(eval_pred.label_ids, predictions, average='weighted')
        }

    def load_and_process_dataset(self):
        """Load and preprocess MNLI dataset"""
        logger.info("Loading MNLI dataset...")
        dataset = load_dataset("glue", "mnli")

        # Rename mismatched validation set for consistency
        dataset = dataset.remove_columns(['idx'])

        logger.info("Preprocessing dataset...")
        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=dataset["train"].column_names
        )

        return tokenized_dataset

    def get_training_args(self, num_training_steps: int) -> TrainingArguments:
        """Configure training arguments"""
        return TrainingArguments(
            output_dir=f"{self.output_dir}/{self.model_name.split('/')[-1]}",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            logging_dir="./logs",
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="tensorboard",
            fp16=True,
            gradient_checkpointing=True,
            gradient_accumulation_steps=2,
            dataloader_num_workers=4,
            # MNLI specific: evaluate on both matched and mismatched dev sets
            eval_steps=0.1,  # Evaluate every 10% of training
        )

    def train(self):
        """Train the Longformer model on MNLI"""
        # Load and process dataset
        dataset = self.load_and_process_dataset()

        # Calculate training steps
        num_training_steps = (
            len(dataset["train"])
            * self.num_epochs
            // (self.batch_size * 2)  # Account for gradient accumulation
        )

        # Initialize trainer
        training_args = self.get_training_args(num_training_steps)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset={
                'matched': dataset["validation_matched"],
                'mismatched': dataset["validation_mismatched"]
            },
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
        )

        # Train and evaluate
        logger.info("Starting training...")
        trainer.train()

        # Save final model
        trainer.save_model()
        logger.info(f"Model saved to {training_args.output_dir}")

        # Final evaluation on both validation sets
        logger.info("Evaluating on matched validation set...")
        matched_results = trainer.evaluate(dataset["validation_matched"])
        logger.info(f"Matched validation results: {matched_results}")

        logger.info("Evaluating on mismatched validation set...")
        mismatched_results = trainer.evaluate(dataset["validation_mismatched"])
        logger.info(f"Mismatched validation results: {mismatched_results}")

        return {
            'matched': matched_results,
            'mismatched': mismatched_results
        }

    def save_predictions(self, output_file: str):
        """Save predictions on test set"""
        # Load test dataset
        test_dataset = load_dataset("glue", "mnli", split="test")

        # Preprocess test dataset
        test_encoded = self.preprocess_function(test_dataset)

        # Get predictions
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(test_encoded).predictions
        predictions = np.argmax(predictions, axis=1)

        # Convert to labels
        predictions = [self.id2label[pred] for pred in predictions]

        # Save predictions
        with open(output_file, 'w') as f:
            f.write("index\tprediction\n")
            for idx, pred in enumerate(predictions):
                f.write(f"{idx}\t{pred}\n")


def main():
    # Initialize trainer
    trainer = LongformerMNLITrainer(
        model_name="allenai/longformer-base-4096",
        batch_size=16,
        max_length=512,
        num_epochs=3,
        learning_rate=2e-5,
        output_dir="./longformer_mnli_models"
    )

    # Train the model
    results = trainer.train()

    # Save test predictions if needed
    # trainer.save_predictions("mnli_predictions.tsv")


if __name__ == "__main__":
    main()
