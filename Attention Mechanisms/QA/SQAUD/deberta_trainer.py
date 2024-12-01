import torch
from datasets import load_dataset
from transformers import (
    DebertaV2TokenizerFast,
    DebertaV2ForQuestionAnswering,
    Trainer,
    TrainingArguments
)
import logging
import sys
from typing import Dict, Any
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for DeBERTa training."""
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 384
    doc_stride: int = 128
    batch_size: int = 16  # DeBERTa can handle larger batches
    learning_rate: float = 1e-5  # Lower learning rate for DeBERTa
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    fp16: bool = True  # Enable mixed precision training
    output_dir: str = field(default_factory=lambda: "./deberta_squad")


class DeBERTaSquadTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize tokenizer
        self.tokenizer = DebertaV2TokenizerFast.from_pretrained(
            config.model_name)
        logger.info(f"Initialized tokenizer: {config.model_name}")

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
            return_offsets_mapping=True
        )

        sample_map = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        # Initialize answer positions
        tokenized["start_positions"] = []
        tokenized["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = examples["answers"][sample_idx]

            # Get sequence IDs to identify context tokens
            sequence_ids = tokenized.sequence_ids(i)

            # Find context boundaries
            context_start = sequence_ids.index(1)
            context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

            # Handle no-answer cases (SQuAD 2.0)
            if len(answer["answer_start"]) == 0:
                tokenized["start_positions"].append(0)
                tokenized["end_positions"].append(0)
                continue

            # Get answer boundaries
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            # Check if the answer is fully within the context
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

        return tokenized

    def train(self, squad_version: str = "1.1"):
        """Train DeBERTa on specified SQuAD version."""
        logger.info(f"Starting DeBERTa training on SQuAD {squad_version}")

        # Load dataset
        dataset_name = "squad_v2" if squad_version == "2.0" else "squad"
        dataset = load_dataset(dataset_name)
        logger.info(f"Loaded {dataset_name} dataset")

        # Preprocess dataset
        tokenized_dataset = dataset.map(
            self.preprocess_data,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Preprocessing dataset",
            num_proc=4  # Parallel processing
        )
        logger.info("Dataset preprocessing completed")

        # Initialize model
        model = DebertaV2ForQuestionAnswering.from_pretrained(
            self.config.model_name)
        model = model.to(self.device)

        # Calculate training steps and warmup steps
        num_training_steps = (
            len(tokenized_dataset["train"])
            // (self.config.batch_size * self.config.gradient_accumulation_steps)
            * self.config.num_epochs
        )
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}_{squad_version}",
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
            logging_dir=f"{self.config.output_dir}_{squad_version}/logs",
            logging_steps=100
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=self.tokenizer
        )

        # Train the model
        logger.info("Starting training...")
        trainer.train()

        # Save the final model
        model_save_path = f"{self.config.output_dir}_{squad_version}_final"
        trainer.save_model(model_save_path)
        logger.info(f"Model saved to {model_save_path}")

        # Final evaluation
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")

        return eval_results


def main():
    # Define configuration
    config = TrainingConfig(
        model_name="microsoft/deberta-v3-base",
        batch_size=16,
        learning_rate=1e-5,
        num_epochs=3,
        gradient_accumulation_steps=2,
        fp16=True
    )

    # Initialize trainer
    trainer = DeBERTaSquadTrainer(config)

    # Train on SQuAD v1.1
    logger.info("Starting SQuAD v1.1 training")
    v1_results = trainer.train(squad_version="1.1")

    # Train on SQuAD v2.0
    logger.info("Starting SQuAD v2.0 training")
    v2_results = trainer.train(squad_version="2.0")

    # Log final results
    logger.info("\nTraining Complete!")
    logger.info(f"SQuAD v1.1 Results: {v1_results}")
    logger.info(f"SQuAD v2.0 Results: {v2_results}")


if __name__ == "__main__":
    main()
