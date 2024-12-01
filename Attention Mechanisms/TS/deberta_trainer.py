import torch
from datasets import load_dataset
import rouge_score
import numpy as np
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from rouge_score import rouge_scorer
import logging
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


class RougeCalculator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

    def compute_rouge_metrics(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute ROUGE scores for a batch of predictions and references"""
        scores = defaultdict(list)

        for pred, ref in zip(predictions, references):
            # Calculate scores for each pair
            score = self.scorer.score(ref, pred)

            # Collect precision, recall, and f1 for each ROUGE type
            for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                scores[f'{rouge_type}_precision'].append(
                    score[rouge_type].precision)
                scores[f'{rouge_type}_recall'].append(score[rouge_type].recall)
                scores[f'{rouge_type}_f1'].append(score[rouge_type].fmeasure)

        # Calculate averages
        results = {}
        for metric, values in scores.items():
            results[metric] = np.mean(values)

        return results


@dataclass
class DatasetConfig:
    """Configuration for each dataset"""
    name: str
    text_column: str
    label_column: str
    subset: Optional[str] = None
    max_samples: Optional[int] = None
    label_map: Optional[Dict] = None


class DeBERTaScientificTrainer:
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        max_length: int = 512,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        output_dir: str = "./deberta_scientific_models"
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.output_dir = output_dir

        # Initialize ROUGE calculator
        self.rouge_calculator = RougeCalculator()

        # Dataset configurations
        self.dataset_configs = [
            DatasetConfig(
                name="arxiv_dataset",
                text_column="abstract",
                label_column="categories",
                max_samples=100000
            ),
            DatasetConfig(
                name="pubmed_dataset",
                text_column="abstract",
                label_column="category",
                max_samples=100000
            ),
            DatasetConfig(
                name="big_patent",
                text_column="description",
                label_column="section_id",
                subset="a",
                max_samples=100000
            )
        ]

        # Initialize tokenizer and model
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        self.label_set = self._get_combined_label_set()
        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.label_set)
        )

    def _get_combined_label_set(self):
        """Get combined set of unique labels from all datasets"""
        label_set = set()

        for config in self.dataset_configs:
            try:
                dataset = load_dataset(
                    config.name,
                    config.subset,
                    split="train[:1000]"
                )
                labels = dataset[config.label_column]

                if isinstance(labels[0], str):
                    label_set.update(labels)
                elif isinstance(labels[0], list):
                    for label_list in labels:
                        label_set.update(label_list)
            except Exception as e:
                logger.error(f"Error loading labels from {
                             config.name}: {str(e)}")

        return sorted(list(label_set))

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict:
        """Compute ROUGE metrics for evaluation"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        references = eval_pred.label_ids

        # Convert indices to text for ROUGE calculation
        pred_texts = [self.label_set[pred] for pred in predictions]
        ref_texts = [self.label_set[ref] for ref in references]

        # Calculate ROUGE scores
        rouge_scores = self.rouge_calculator.compute_rouge_metrics(
            pred_texts, ref_texts)

        # Add summary metrics
        results = {
            'rouge1_f1': rouge_scores['rouge1_f1'],
            'rouge2_f1': rouge_scores['rouge2_f1'],
            'rougeL_f1': rouge_scores['rougeL_f1'],
            # Add individual precision and recall scores
            'rouge1_precision': rouge_scores['rouge1_precision'],
            'rouge1_recall': rouge_scores['rouge1_recall'],
            'rouge2_precision': rouge_scores['rouge2_precision'],
            'rouge2_recall': rouge_scores['rouge2_recall'],
            'rougeL_precision': rouge_scores['rougeL_precision'],
            'rougeL_recall': rouge_scores['rougeL_recall']
        }

        return results

    def preprocess_function(self, examples, config):
        """Prepare examples for the model"""
        # Handle text preprocessing
        texts = examples[config.text_column]
        if isinstance(texts[0], list):
            texts = [' '.join(text) for text in texts]

        # Tokenize texts
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )

        # Process labels
        labels = examples[config.label_column]
        if isinstance(labels[0], list):
            # Multi-label case
            label_matrix = np.zeros((len(labels), len(self.label_set)))
            for i, label_list in enumerate(labels):
                for label in label_list:
                    if label in self.label_set:
                        label_matrix[i, self.label_set.index(label)] = 1
            tokenized["labels"] = label_matrix.tolist()
        else:
            # Single-label case
            tokenized["labels"] = [
                self.label_set.index(label) if label in self.label_set else -1
                for label in labels
            ]

        return tokenized

    def load_and_process_datasets(self):
        """Load and preprocess all datasets"""
        processed_datasets = defaultdict(list)

        for config in self.dataset_configs:
            logger.info(f"Processing dataset: {config.name}")
            try:
                # Load dataset
                dataset = load_dataset(
                    config.name,
                    config.subset,
                    split='train' if config.max_samples is None
                    else f'train[:{config.max_samples}]'
                )

                # Preprocess dataset
                tokenized = dataset.map(
                    lambda x: self.preprocess_function(x, config),
                    batched=True,
                    num_proc=4,
                    remove_columns=dataset.column_names
                )

                processed_datasets['train'].append(tokenized)

                # Load validation set
                try:
                    val_dataset = load_dataset(
                        config.name,
                        config.subset,
                        split='validation'
                    )
                    val_tokenized = val_dataset.map(
                        lambda x: self.preprocess_function(x, config),
                        batched=True,
                        num_proc=4,
                        remove_columns=val_dataset.column_names
                    )
                    processed_datasets['validation'].append(val_tokenized)
                except Exception as e:
                    logger.warning(f"No validation set for {
                                   config.name}: {str(e)}")

            except Exception as e:
                logger.error(f"Error processing {config.name}: {str(e)}")
                continue

        # Combine datasets
        from datasets import concatenate_datasets
        combined_datasets = {
            split: concatenate_datasets(datasets) if datasets else None
            for split, datasets in processed_datasets.items()
        }

        return combined_datasets

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
            metric_for_best_model="rougeL_f1",  # Use ROUGE-L F1 as primary metric
            greater_is_better=True,
            report_to="tensorboard",
            fp16=True,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            dataloader_num_workers=4,
        )

    def train(self):
        """Train the model on all datasets"""
        # Load and process datasets
        logger.info("Loading and processing datasets...")
        datasets = self.load_and_process_datasets()

        if not datasets['train']:
            raise ValueError("No training data available")

        # Calculate training steps
        num_training_steps = (
            len(datasets['train'])
            * self.num_epochs
            // (self.batch_size * 4)
        )

        # Initialize trainer
        training_args = self.get_training_args(num_training_steps)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets.get('validation'),
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
        if datasets.get('validation'):
            eval_results = trainer.evaluate()
            logger.info(f"Final evaluation results: {eval_results}")
            return eval_results

        return None


def main():
    # Initialize trainer
    trainer = DeBERTaScientificTrainer(
        model_name="microsoft/deberta-v3-base",
        batch_size=8,
        max_length=512,
        num_epochs=3,
        learning_rate=2e-5,
        output_dir="./deberta_scientific_models"
    )

    # Train the model
    trainer.train()


if __name__ == "__main__":
    main()
