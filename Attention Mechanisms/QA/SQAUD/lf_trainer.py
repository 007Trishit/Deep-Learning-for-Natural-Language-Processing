#!/usr/bin/env python3
import torch
import logging
import sys
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datasets import load_dataset
from transformers import (
    LongformerTokenizerFast,
    LongformerForQuestionAnswering,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
import numpy as np
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class LongformerConfig:
    """Configuration for Longformer training."""
    model_name: str = "allenai/longformer-base-4096"
    max_length: int = 4096
    doc_stride: int = 512
    batch_size: int = 4
    learning_rate: float = 3e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    fp16: bool = False  # Disabled to prevent memory issues
    attention_window: int = 512
    output_dir: str = field(default_factory=lambda: "./longformer_squad")
    dataloader_num_workers: int = 2
    squad_version: str = "1.1"


class LongformerSquadTrainer:
    def __init__(self, config: LongformerConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize tokenizer
        self.tokenizer = LongformerTokenizerFast.from_pretrained(
            config.model_name)
        logger.info(f"Initialized tokenizer: {config.model_name}")

        # Initialize model
        self.model = LongformerForQuestionAnswering.from_pretrained(
            config.model_name,
            attention_window=config.attention_window
        )
        self.model.to(self.device)

    def preprocess_data(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize and prepare the dataset examples."""
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]

        # Tokenize inputs
        tokenized = self.tokenizer(
            questions,
            contexts,
            max_length=self.config.max_length,
            stride=self.config.doc_stride,
            truncation="only_second",
            padding="max_length",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True
        )

        # Handle global attention mask for question tokens
        global_attention_mask = torch.zeros_like(
            torch.tensor(tokenized["attention_mask"]))

        sample_map = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        tokenized["start_positions"] = []
        tokenized["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = examples["answers"][sample_idx]

            # Get sequence IDs safely
            sequence_ids = tokenized.sequence_ids(i)
            if sequence_ids is None:
                tokenized["start_positions"].append(0)
                tokenized["end_positions"].append(0)
                continue

            # Set global attention on question tokens and [CLS] token
            question_indices = [j for j, seq_id in enumerate(
                sequence_ids) if seq_id == 0]
            global_attention_mask[i, 0] = 1  # [CLS] token
            if question_indices:
                global_attention_mask[i, question_indices] = 1

            try:
                context_start = sequence_ids.index(1)
                context_end = len(sequence_ids) - 1 - \
                    sequence_ids[::-1].index(1)
            except ValueError:
                tokenized["start_positions"].append(0)
                tokenized["end_positions"].append(0)
                continue

            # Handle no-answer cases
            if len(answer["answer_start"]) == 0:
                tokenized["start_positions"].append(0)
                tokenized["end_positions"].append(0)
                continue

            # Get answer boundaries
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            if offsets[context_start][0] > start_char or offsets[context_end][1] < end_char:
                tokenized["start_positions"].append(0)
                tokenized["end_positions"].append(0)
                continue

            # Find token positions
            token_start = token_end = 0

            # Find start position
            for idx in range(context_start, context_end + 1):
                if offsets[idx][0] <= start_char <= offsets[idx][1]:
                    token_start = idx
                    break

            # Find end position
            for idx in range(context_end, context_start - 1, -1):
                if offsets[idx][0] <= end_char <= offsets[idx][1]:
                    token_end = idx
                    break

            tokenized["start_positions"].append(token_start)
            tokenized["end_positions"].append(token_end)

        tokenized["global_attention_mask"] = global_attention_mask.tolist()
        return tokenized

    def compute_metrics(self, pred: EvalPrediction) -> Dict[str, float]:
        """Compute evaluation metrics."""
        start_logits, end_logits = pred.predictions

        # Convert predictions to probabilities
        start_probs = torch.nn.functional.softmax(
            torch.tensor(start_logits), dim=-1)
        end_probs = torch.nn.functional.softmax(
            torch.tensor(end_logits), dim=-1)

        # Get the most likely positions
        predicted_starts = np.argmax(start_logits, axis=1)
        predicted_ends = np.argmax(end_logits, axis=1)

        # Calculate confidence scores
        confidence_scores = []
        for i in range(len(predicted_starts)):
            if predicted_ends[i] >= predicted_starts[i]:
                confidence = float(
                    start_probs[i, predicted_starts[i]] * end_probs[i, predicted_ends[i]])
                confidence_scores.append(confidence)
            else:
                confidence_scores.append(0.0)

        metrics = {
            "confidence_mean": np.mean(confidence_scores),
            "confidence_std": np.std(confidence_scores),
        }

        return metrics

    def train(self) -> Dict[str, float]:
        """Train the model on specified SQuAD version."""
        logger.info(
            f"Starting Longformer training on SQuAD {self.config.squad_version}")

        # Load dataset
        dataset_name = "squad_v2" if self.config.squad_version == "2.0" else "squad"
        dataset = load_dataset(dataset_name)
        logger.info(f"Loaded {dataset_name} dataset")

        # Preprocess dataset
        tokenized_dataset = dataset.map(
            self.preprocess_data,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Preprocessing dataset",
            num_proc=self.config.dataloader_num_workers
        )
        logger.info("Dataset preprocessing completed")

        # Calculate steps
        num_training_steps = (
            len(tokenized_dataset["train"])
            // (self.config.batch_size * self.config.gradient_accumulation_steps)
            * self.config.num_epochs
        )
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}_{self.config.squad_version}",
            evaluation_strategy="steps",
            eval_steps=500,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            warmup_steps=warmup_steps,
            fp16=self.config.fp16,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="tensorboard",
            logging_dir=f"{self.config.output_dir}_{self.config.squad_version}/logs",
            logging_steps=100,
            dataloader_num_workers=self.config.dataloader_num_workers
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        # Train the model
        logger.info("Starting training...")
        train_result = trainer.train()

        # Save the final model
        model_save_path = f"{self.config.output_dir}_{self.config.squad_version}_final"
        trainer.save_model(model_save_path)
        logger.info(f"Model saved to {model_save_path}")

        # Final evaluation
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")

        return eval_results


def parse_args():
    parser = argparse.ArgumentParser(description="Train Longformer on SQuAD")
    parser.add_argument("--squad_version", type=str,
                        default="1.1", choices=["1.1", "2.0"])
    parser.add_argument("--output_dir", type=str, default="./longformer_squad")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--model_name", type=str,
                        default="allenai/longformer-base-4096")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize configuration
    config = LongformerConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        squad_version=args.squad_version
    )

    try:
        # Initialize trainer
        trainer = LongformerSquadTrainer(config)

        # Train the model
        results = trainer.train()

        # Log results
        logger.info("\nTraining completed successfully!")
        logger.info(f"Final Results: {results}")

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
