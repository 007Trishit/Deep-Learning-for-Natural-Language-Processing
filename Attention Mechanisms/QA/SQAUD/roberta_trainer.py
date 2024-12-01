import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForQuestionAnswering,
    Trainer,
    TrainingArguments
)
import logging
import sys
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class SQuADTrainer:
    def __init__(
        self,
        model_name: str = "roberta-base",
        max_length: int = 384,
        doc_stride: int = 128,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay

        # Initialize tokenizer and model
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

    def preprocess_data(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize and prepare the dataset examples."""
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]

        # Tokenize inputs
        tokenized = self.tokenizer(
            questions,
            contexts,
            max_length=self.max_length,
            stride=self.doc_stride,
            truncation="only_second",
            padding="max_length",
            return_overflowing_tokens=True,
            return_offsets_mapping=True
        )

        # Extract sample mapping and offset mapping
        sample_map = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        # Initialize answer positions
        tokenized["start_positions"] = []
        tokenized["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # Get the sample index for the current span
            sample_idx = sample_map[i]
            answer = examples["answers"][sample_idx]

            # Get sequence IDs to identify context tokens
            sequence_ids = tokenized.sequence_ids(i)
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

    def train(self, squad_version: str = "1.1") -> None:
        """Train the model on specified SQuAD version."""
        logger.info(f"Starting training on SQuAD {squad_version}")

        # Load dataset
        dataset_name = "squad_v2" if squad_version == "2.0" else "squad"
        dataset = load_dataset(dataset_name)

        # Preprocess dataset
        tokenized_dataset = dataset.map(
            self.preprocess_data,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Preprocessing dataset"
        )

        # Initialize model
        model = RobertaForQuestionAnswering.from_pretrained(self.model_name)

        # Calculate warmup steps
        num_training_steps = (
            len(tokenized_dataset["train"])
            // self.batch_size
            * self.num_epochs
        )
        warmup_steps = int(num_training_steps * self.warmup_ratio)

        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=f"./roberta_squad{squad_version}",
            evaluation_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=self.weight_decay,
            warmup_steps=warmup_steps,
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="tensorboard"
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

        # Save the model
        model_save_path = f"./roberta_squad{squad_version}_final"
        trainer.save_model(model_save_path)
        logger.info(f"Model saved to {model_save_path}")

        # Evaluate the model
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")


def main():
    # Initialize trainer
    trainer = SQuADTrainer(
        model_name="roberta-base",
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=3
    )

    # Train on SQuAD v1.1
    trainer.train(squad_version="1.1")

    # Train on SQuAD v2.0
    trainer.train(squad_version="2.0")


if __name__ == "__main__":
    main()
