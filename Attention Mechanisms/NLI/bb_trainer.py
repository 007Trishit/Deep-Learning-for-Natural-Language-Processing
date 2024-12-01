import torch
from datasets import load_dataset
from transformers import (
    BigBirdTokenizer,
    BigBirdForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
import numpy as np
import logging
import sys
from typing import Dict, List
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


class BigBirdMNLITrainer:
    def __init__(
        self,
        model_name: str = "google/bigbird-roberta-base",
        max_length: int = 1024,  # BigBird can handle longer sequences
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        output_dir: str = "./bigbird_mnli_models",
        block_size: int = 64,  # BigBird specific parameter
        num_random_blocks: int = 3  # BigBird specific parameter
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.output_dir = output_dir
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks

        # Initialize tokenizer
        self.tokenizer = BigBirdTokenizer.from_pretrained(model_name)

        # Initialize model with classification head
        self.model = BigBirdForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,  # MNLI has 3 classes
            block_size=block_size,
            num_random_blocks=num_random_blocks,
            attention_type="block_sparse"  # Use block sparse attention
        )

        # Label mapping
        self.label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def preprocess_function(self, examples):
        """Prepare MNLI examples for BigBird"""
        # Tokenize premises and hypotheses
        tokenized = self.tokenizer(
            examples['premise'],
            examples['hypothesis'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )

        # Add labels
        tokenized["labels"] = [
            self.label2id[label] for label in examples['label']
        ]

        return tokenized

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict:
        """Compute metrics for evaluation"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(labels, predictions)

        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in self.id2label.items():
            true_pos = conf_matrix[i, i]
            false_pos = conf_matrix[:, i].sum() - true_pos
            false_neg = conf_matrix[i, :].sum() - true_pos

            precision = true_pos / \
                (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / \
                (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * (precision * recall) / (precision +
                                             recall) if (precision + recall) > 0 else 0

            per_class_metrics.update({
                f'{class_name}_precision': precision,
                f'{class_name}_recall': recall,
                f'{class_name}_f1': f1
            })

        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            **per_class_metrics
        }

    def load_and_process_dataset(self):
        """Load and preprocess MNLI dataset"""
        logger.info("Loading MNLI dataset...")
        dataset = load_dataset("glue", "mnli")

        # Remove unnecessary columns
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
            evaluation_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="tensorboard",
            fp16=True,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            dataloader_num_workers=4,
            # Additional BigBird optimizations
            ddp_find_unused_parameters=False,
            sharded_ddp="simple"  # Enable sharded training
        )

    def train(self):
        """Train the BigBird model on MNLI"""
        # Load and process dataset
        dataset = self.load_and_process_dataset()

        # Calculate training steps
        num_training_steps = (
            len(dataset["train"])
            * self.num_epochs
            // (self.batch_size * 4)  # Account for gradient accumulation
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

        # Final evaluation
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

    def predict(self, text_pairs: List[Dict[str, str]]) -> List[str]:
        """Make predictions on new text pairs"""
        inputs = [
            (pair['premise'], pair['hypothesis'])
            for pair in text_pairs
        ]

        # Tokenize inputs
        features = self.tokenizer(
            [pair[0] for pair in inputs],
            [pair[1] for pair in inputs],
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        features = {k: v.to(device) for k, v in features.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**features)
            predictions = torch.argmax(outputs.logits, dim=1)

        # Convert to labels
        return [self.id2label[pred.item()] for pred in predictions]


def main():
    # Initialize trainer
    trainer = BigBirdMNLITrainer(
        model_name="google/bigbird-roberta-base",
        batch_size=8,
        max_length=1024,
        num_epochs=3,
        learning_rate=2e-5,
        output_dir="./bigbird_mnli_models",
        block_size=64,
        num_random_blocks=3
    )

    # Train the model
    results = trainer.train()

    # Example prediction
    test_pairs = [
        {
            'premise': "The company's revenue grew by 50% last year.",
            'hypothesis': "The company performed well financially."
        },
        {
            'premise': "The cat is sleeping on the mat.",
            'hypothesis': "The dog is running in the park."
        }
    ]
    predictions = trainer.predict(test_pairs)
    logger.info(f"Test predictions: {predictions}")


if __name__ == "__main__":
    main()
