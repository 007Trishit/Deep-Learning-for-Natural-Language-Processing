import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    RobertaTokenizerFast,
    RobertaForQuestionAnswering,
    DebertaV2TokenizerFast,
    DebertaV2ForQuestionAnswering
)
from typing import Optional, Tuple
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class SQuADTrainer:
    def __init__(
        self,
        model_name: str,
        squad_version: str = "1.1",
        max_length: int = 384,
        doc_stride: int = 128,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        output_dir: str = "./saved_models"
    ):
        self.model_name = model_name
        self.squad_version = squad_version
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.output_dir = output_dir

        # Load appropriate model and tokenizer
        if "roberta" in model_name.lower():
            self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
            self.model = RobertaForQuestionAnswering.from_pretrained(
                model_name)
        elif "deberta" in model_name.lower():
            self.tokenizer = DebertaV2TokenizerFast.from_pretrained(model_name)
            self.model = DebertaV2ForQuestionAnswering.from_pretrained(
                model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

    def load_dataset(self) -> Tuple[dict, dict]:
        """Load SQuAD dataset based on version."""
        dataset_name = "squad_v2" if self.squad_version == "2.0" else "squad"
        logger.info(f"Loading {dataset_name} dataset...")
        return load_dataset(dataset_name)

    def preprocess_function(self, examples):
        """Tokenize and prepare the dataset."""
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]

            # Handle no-answer case for SQuAD 2.0
            if len(answer["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find context boundaries
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Find token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def prepare_dataset(self, dataset):
        """Prepare the dataset for training."""
        logger.info("Preprocessing dataset...")
        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        return tokenized_dataset

    def get_training_args(self):
        """Configure training arguments."""
        return TrainingArguments(
            output_dir=f"{self.output_dir}/{self.squad_version}/{self.model_name.split('/')[-1]}",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            logging_dir="./logs",
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="tensorboard"
        )

    def train(self):
        """Train the model on SQuAD."""
        # Load dataset
        dataset = self.load_dataset()
        tokenized_dataset = self.prepare_dataset(dataset)

        # Initialize trainer
        training_args = self.get_training_args()
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
        )

        # Train and evaluate
        logger.info("Starting training...")
        trainer.train()

        # Save the final model
        trainer.save_model()
        logger.info(f"Model saved to {training_args.output_dir}")

        # Final evaluation
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")
        return eval_results


def main():
    # Train RoBERTa on SQuAD 1.1
    roberta_squad1 = SQuADTrainer(
        model_name="roberta-base",
        squad_version="1.1",
        batch_size=8,
        output_dir="./full_models"
    )
    roberta_squad1.train()

    # Train RoBERTa on SQuAD 2.0
    roberta_squad2 = SQuADTrainer(
        model_name="roberta-base",
        squad_version="2.0",
        batch_size=8,
        output_dir="./full_models"
    )
    roberta_squad2.train()

    # Train DeBERTa on SQuAD 1.1
    deberta_squad1 = SQuADTrainer(
        model_name="microsoft/deberta-v3-base",
        squad_version="1.1",
        batch_size=16,  # DeBERTa can handle larger batch sizes
        output_dir="./full_models"
    )
    deberta_squad1.train()

    # Train DeBERTa on SQuAD 2.0
    deberta_squad2 = SQuADTrainer(
        model_name="microsoft/deberta-v3-base",
        squad_version="2.0",
        batch_size=16,
        output_dir="./full_models"
    )
    deberta_squad2.train()


if __name__ == "__main__":
    main()
