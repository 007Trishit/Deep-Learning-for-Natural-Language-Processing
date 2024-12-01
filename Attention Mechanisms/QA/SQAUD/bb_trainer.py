import torch
from datasets import load_dataset
from transformers import (
    BigBirdTokenizerFast,
    BigBirdForQuestionAnswering,
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
class BigBirdConfig:
    """Configuration for BigBird training."""
    model_name: str = "google/bigbird-roberta-base"
    max_length: int = 2048  # BigBird can handle longer sequences
    doc_stride: int = 512   # Larger stride for longer sequences
    batch_size: int = 16    # Smaller batch size due to memory requirements
    learning_rate: float = 3e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    block_size: int = 64    # BigBird-specific block size
    num_random_blocks: int = 3  # BigBird-specific random blocks
    output_dir: str = field(default_factory=lambda: "./bigbird_squad")


class BigBirdSquadTrainer:
    def __init__(self, config: BigBirdConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize tokenizer
        self.tokenizer = BigBirdTokenizerFast.from_pretrained(config.model_name)
        logger.info(f"Initialized tokenizer: {config.model_name}")

    def preprocess_data(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize and prepare the dataset examples."""
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]

        # Tokenize inputs with BigBird's longer sequence capability
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

            # Get sequence IDs
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

            # Check if answer is within context
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
        """Train BigBird on specified SQuAD version."""
        logger.info(f"Starting BigBird training on SQuAD {squad_version}")

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
            num_proc=4
        )
        logger.info("Dataset preprocessing completed")

        # Initialize model with BigBird-specific configuration
        model = BigBirdForQuestionAnswering.from_pretrained(
            self.config.model_name,
            block_size=self.config.block_size,
            num_random_blocks=self.config.num_random_blocks
        )
        model = model.to(self.device)

        # Calculate steps
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
            logging_steps=100,
            # BigBird-specific settings
            max_grad_norm=1.0,
            gradient_checkpointing=True  # Enable for memory efficiency
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
    config = BigBirdConfig(
        model_name="google/bigbird-roberta-base",
        batch_size=16,
        learning_rate=3e-5,
        num_epochs=3,
        gradient_accumulation_steps=4,
        block_size=64,
        num_random_blocks=3
    )

    # Initialize trainer
    trainer = BigBirdSquadTrainer(config)

    try:
        # Train on SQuAD v1.1
        logger.info("Starting SQuAD v1.1 training")
        v1_results = trainer.train(squad_version="1.1")

        # Train on SQuAD v2.0
        # logger.info("Starting SQuAD v2.0 training")
        # v2_results = trainer.train(squad_version="2.0")

        # Log final results
        logger.info("\nTraining Complete!")
        logger.info(f"SQuAD v1.1 Results: {v1_results}")
        # logger.info(f"SQuAD v2.0 Results: {v2_results}")

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
