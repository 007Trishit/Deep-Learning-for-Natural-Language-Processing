import torch
from datasets import load_dataset
from transformers import (
    DebertaV2TokenizerFast,
    DebertaV2ForQuestionAnswering,
    Trainer,
    TrainingArguments
)
from typing import Optional, Tuple, List, Dict
import logging
import sys
from dataclasses import dataclass
from datasets import concatenate_datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    name: str
    subset: Optional[str] = None
    question_column: str = "question"
    context_column: str = "context"
    answer_start_column: str = "answer_start"
    answer_text_column: str = "answer_text"

    def get_dataset_loading_args(self) -> Dict:
        args = {"path": self.name}
        if self.subset:
            args["name"] = self.subset
        return args


class MultiDatasetQATrainer:
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        max_length: int = 512,  # Increased for complex questions
        doc_stride: int = 128,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        output_dir: str = "./saved_models",
        dataset_configs: Optional[List[DatasetConfig]] = None
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.output_dir = output_dir

        # Default dataset configurations
        self.dataset_configs = dataset_configs or [
            DatasetConfig(
                name="natural_questions",
                subset="default",
                question_column="question",
                context_column="context",
                answer_start_column="answer_start",
                answer_text_column="answer_text"
            ),
            DatasetConfig(
                name="hotpot_qa",
                subset="distractor",
                question_column="question",
                context_column="context",
                answer_start_column="answer_start",
                answer_text_column="answer_text"
            ),
            DatasetConfig(
                name="trivia_qa",
                subset="rc",
                question_column="question",
                context_column="context",
                answer_start_column="answer_start",
                answer_text_column="answer_text"
            ),
            DatasetConfig(
                name="wiki_hop",
                subset="original",
                question_column="question",
                context_column="context",
                answer_start_column="answer_start",
                answer_text_column="answer_text"
            )
        ]

        # Initialize model and tokenizer
        self.tokenizer = DebertaV2TokenizerFast.from_pretrained(model_name)
        self.model = DebertaV2ForQuestionAnswering.from_pretrained(model_name)

    def load_and_prepare_dataset(self, config: DatasetConfig):
        """Load and prepare a single dataset."""
        logger.info(f"Loading dataset: {config.name}")

        try:
            dataset = load_dataset(**config.get_dataset_loading_args())

            # Preprocess function specific to this dataset
            def preprocess_function(examples):
                questions = [q.strip()
                             for q in examples[config.question_column]]
                contexts = examples[config.context_column]

                # Tokenize inputs
                inputs = self.tokenizer(
                    questions,
                    contexts,
                    max_length=self.max_length,
                    truncation="only_second",
                    stride=self.doc_stride,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding="max_length",
                )

                # Process answer spans
                offset_mapping = inputs.pop("offset_mapping")
                sample_map = inputs.pop("overflow_to_sample_mapping")
                start_positions = []
                end_positions = []

                for i, offset in enumerate(offset_mapping):
                    sample_idx = sample_map[i]
                    answer_starts = examples[config.answer_start_column][sample_idx]
                    answer_texts = examples[config.answer_text_column][sample_idx]

                    # Handle cases where answer might be in different format
                    if isinstance(answer_starts, list):
                        answer_start = answer_starts[0] if answer_starts else 0
                        answer_text = answer_texts[0] if answer_texts else ""
                    else:
                        answer_start = answer_starts
                        answer_text = answer_texts

                    start_char = answer_start
                    end_char = start_char + len(answer_text)

                    # Find token positions
                    sequence_ids = inputs.sequence_ids(i)
                    context_start = sequence_ids.index(1)
                    context_end = len(sequence_ids) - 1 - \
                        sequence_ids[::-1].index(1)

                    if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                        start_positions.append(0)
                        end_positions.append(0)
                    else:
                        # Find start position
                        idx = context_start
                        while idx <= context_end and offset[idx][0] <= start_char:
                            idx += 1
                        start_positions.append(idx - 1)

                        # Find end position
                        idx = context_end
                        while idx >= context_start and offset[idx][1] >= end_char:
                            idx -= 1
                        end_positions.append(idx + 1)

                inputs["start_positions"] = start_positions
                inputs["end_positions"] = end_positions
                return inputs

            # Apply preprocessing
            tokenized_dataset = dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=dataset["train"].column_names,
            )

            return tokenized_dataset

        except Exception as e:
            logger.error(f"Error processing dataset {config.name}: {str(e)}")
            return None

    def prepare_combined_dataset(self):
        """Prepare and combine all datasets."""
        all_train_datasets = []
        all_eval_datasets = []

        for config in self.dataset_configs:
            dataset = self.load_and_prepare_dataset(config)
            if dataset:
                all_train_datasets.append(dataset["train"])
                all_eval_datasets.append(dataset["validation"])

        # Combine datasets
        if all_train_datasets and all_eval_datasets:
            combined_train = concatenate_datasets(all_train_datasets)
            combined_eval = concatenate_datasets(all_eval_datasets)
            return combined_train, combined_eval
        else:
            raise ValueError("No datasets were successfully processed")

    def get_training_args(self):
        """Configure training arguments."""
        return TrainingArguments(
            output_dir=f"{
                self.output_dir}/multi_qa/{self.model_name.split('/')[-1]}",
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
            report_to="tensorboard",
            gradient_accumulation_steps=4,  # Added for stability with larger datasets
            fp16=True,  # Enable mixed precision training
            gradient_checkpointing=True  # Enable gradient checkpointing for memory efficiency
        )

    def train(self):
        """Train the model on all datasets."""
        # Prepare combined dataset
        logger.info("Preparing combined dataset...")
        train_dataset, eval_dataset = self.prepare_combined_dataset()

        # Initialize trainer
        training_args = self.get_training_args()
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
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
    trainer = MultiDatasetQATrainer(
        model_name="microsoft/deberta-v3-base",
        batch_size=4,  # Reduced batch size due to longer sequences
        max_length=512,  # Increased for complex questions
        num_epochs=3,
        output_dir="./multi_qa_models"
    )
    trainer.train()


if __name__ == "__main__":
    main()
