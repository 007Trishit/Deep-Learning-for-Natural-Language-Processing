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
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


@dataclass
class HotpotExample:
    """Class to hold processed HotpotQA examples"""
    question: str
    context: str
    answer: str
    answer_start: int
    supporting_facts: List[Dict]


class HotpotQATrainer:
    def __init__(
        self,
        model_name: str = "roberta-base",
        max_length: int = 512,  # Longer sequence length for multi-hop QA
        doc_stride: int = 128,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        output_dir: str = "./hotpot_qa_models"
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.output_dir = output_dir

        # Initialize tokenizer and model
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        self.model = RobertaForQuestionAnswering.from_pretrained(model_name)

    def prepare_hotpot_features(self, example: Dict) -> HotpotExample:
        """
        Process HotpotQA example to create concatenated context from supporting facts
        """
        # Extract supporting facts and their content
        supporting_facts = []
        context_parts = []

        # Keep track of character offsets for answer span
        current_offset = 0
        answer_start = -1

        # Process each context and supporting fact
        for title, sentences in example['context']:
            for sent_idx, sentence in enumerate(sentences):
                is_supporting = any(
                    fact[0] == title and fact[1] == sent_idx
                    for fact in example['supporting_facts']
                )

                if is_supporting:
                    supporting_facts.append({
                        'title': title,
                        'sent_idx': sent_idx,
                        'content': sentence
                    })

                    # Add sentence to context
                    if context_parts:
                        context_parts.append(" ")
                        current_offset += 1
                    context_parts.append(sentence)

                    # Check if this sentence contains the answer
                    if example['answer'] in sentence:
                        local_start = sentence.find(example['answer'])
                        if answer_start == -1:  # Only take first occurrence
                            answer_start = current_offset + local_start

                    current_offset += len(sentence)

        # Combine all supporting facts into single context
        full_context = "".join(context_parts)

        return HotpotExample(
            question=example['question'],
            context=full_context,
            answer=example['answer'],
            answer_start=answer_start,
            supporting_facts=supporting_facts
        )

    def preprocess_function(self, examples: Dict) -> Dict:
        """Convert raw examples to model inputs"""
        processed_examples = []

        # Process each example
        for idx in range(len(examples['question'])):
            example = {key: examples[key][idx] for key in examples.keys()}
            processed = self.prepare_hotpot_features(example)
            processed_examples.append(processed)

        # Tokenize questions and contexts
        questions = [ex.question for ex in processed_examples]
        contexts = [ex.context for ex in processed_examples]

        # Tokenize inputs
        tokenized = self.tokenizer(
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
        offset_mapping = tokenized.pop("offset_mapping")
        sample_map = tokenized.pop("overflow_to_sample_mapping")
        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            processed_example = processed_examples[sample_idx]

            # Get answer span
            answer_start = processed_example.answer_start
            answer_end = answer_start + len(processed_example.answer)

            # Find token positions
            sequence_ids = tokenized.sequence_ids(i)
            context_start = sequence_ids.index(1)
            context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

            # If the answer is not fully inside the context, label is (0, 0)
            if offsets[context_start][0] > answer_start or offsets[context_end][1] < answer_end:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Find start position
                idx = context_start
                while idx <= context_end and offsets[idx][0] <= answer_start:
                    idx += 1
                start_positions.append(idx - 1)

                # Find end position
                idx = context_end
                while idx >= context_start and offsets[idx][1] >= answer_end:
                    idx -= 1
                end_positions.append(idx + 1)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized

    def load_and_process_dataset(self):
        """Load and preprocess HotpotQA dataset"""
        logger.info("Loading HotpotQA dataset...")
        dataset = load_dataset("hotpot_qa", "distractor")

        logger.info("Preprocessing dataset...")
        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=4  # Parallel processing
        )

        return tokenized_dataset

    def get_training_args(self, num_training_steps: int) -> TrainingArguments:
        """Configure training arguments"""
        return TrainingArguments(
            output_dir=f"{self.output_dir}/{self.model_name.split('/')[-1]}",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            logging_dir="./logs",
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="tensorboard",
            fp16=True,  # Mixed precision training
            gradient_checkpointing=True,  # Memory optimization
            gradient_accumulation_steps=4  # Stability for longer sequences
        )

    def train(self):
        """Train the model on HotpotQA"""
        # Load and process dataset
        dataset = self.load_and_process_dataset()

        # Calculate number of training steps
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
            eval_dataset=dataset["validation"],
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
    # Initialize trainer with RoBERTa
    trainer = HotpotQATrainer(
        model_name="roberta-base",
        batch_size=4,  # Smaller batch size for longer sequences
        max_length=512,  # Increased for multi-hop questions
        num_epochs=3,
        learning_rate=2e-5,
        output_dir="./hotpot_qa_models"
    )

    # Train the model
    trainer.train()


if __name__ == "__main__":
    main()
