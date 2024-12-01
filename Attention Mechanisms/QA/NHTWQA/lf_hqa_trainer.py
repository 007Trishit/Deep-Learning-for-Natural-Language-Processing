import torch
from datasets import load_dataset
from transformers import (
    LongformerTokenizerFast,
    LongformerForQuestionAnswering,
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
    title_to_text: Dict[str, str]  # Maps titles to their full text


class LongformerHotpotTrainer:
    def __init__(
        self,
        model_name: str = "allenai/longformer-base-4096",
        max_length: int = 4096,  # Longformer can handle much longer sequences
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        output_dir: str = "./longformer_hotpot_models",
        attention_window: int = 512  # Longformer specific parameter
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

        # Initialize tokenizer and model
        self.tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
        self.model = LongformerForQuestionAnswering.from_pretrained(model_name)

        # Set global attention on question tokens
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<question>']})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def prepare_hotpot_features(self, example: Dict) -> HotpotExample:
        """Process HotpotQA example with Longformer-specific handling"""
        # Initialize containers
        title_to_text = {}
        supporting_facts = []
        context_parts = []
        current_offset = 0
        answer_start = -1

        # Process context and supporting facts
        for title, sentences in example['context']:
            # Build full text for each title
            title_text = ' '.join(sentences)
            title_to_text[title] = title_text

            # Process supporting facts
            for sent_idx, sentence in enumerate(sentences):
                is_supporting = any(
                    fact[0] == title and fact[1] == sent_idx
                    for fact in example['supporting_facts']
                )

                if is_supporting:
                    supporting_facts.append({
                        'title': title,
                        'sent_idx': sent_idx,
                        'content': sentence,
                        'offset': current_offset
                    })

                    # Add sentence to context
                    if context_parts:
                        context_parts.append(" ")
                        current_offset += 1
                    context_parts.append(sentence)

                    # Check for answer span
                    if example['answer'] in sentence:
                        local_start = sentence.find(example['answer'])
                        if answer_start == -1:
                            answer_start = current_offset + local_start

                    current_offset += len(sentence)

        # Combine into full context
        full_context = "".join(context_parts)

        return HotpotExample(
            question=example['question'],
            context=full_context,
            answer=example['answer'],
            answer_start=answer_start,
            supporting_facts=supporting_facts,
            title_to_text=title_to_text
        )

    def preprocess_function(self, examples: Dict) -> Dict:
        """Convert raw examples to Longformer inputs"""
        processed_examples = []

        # Process each example
        for idx in range(len(examples['question'])):
            example = {key: examples[key][idx] for key in examples.keys()}
            processed = self.prepare_hotpot_features(example)
            processed_examples.append(processed)

        # Prepare for tokenization
        questions = [f"<question> {
            ex.question} </question>" for ex in processed_examples]
        contexts = [ex.context for ex in processed_examples]

        # Tokenize inputs
        tokenized = self.tokenizer(
            questions,
            contexts,
            max_length=self.max_length,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Prepare global attention mask
        global_attention_mask = torch.zeros(
            (len(tokenized["input_ids"]), self.max_length))

        # Set global attention on question tokens and [SEP] tokens
        for idx, input_ids in enumerate(tokenized["input_ids"]):
            # Find question tokens
            question_tokens = (
                torch.tensor(input_ids) == self.tokenizer.convert_tokens_to_ids(
                    "<question>")
            )
            global_attention_mask[idx][question_tokens] = 1

            # Find [SEP] tokens
            sep_tokens = (
                torch.tensor(input_ids) == self.tokenizer.sep_token_id
            )
            global_attention_mask[idx][sep_tokens] = 1

        tokenized["global_attention_mask"] = global_attention_mask.tolist()

        # Process answer spans
        start_positions = []
        end_positions = []
        offset_mapping = tokenized.pop("offset_mapping")

        for i, offsets in enumerate(offset_mapping):
            processed_example = processed_examples[i]

            # Get answer span
            answer_start = processed_example.answer_start
            answer_end = answer_start + len(processed_example.answer)

            # Find token positions
            sequence_ids = tokenized.sequence_ids(i)
            context_start = sequence_ids.index(1)
            context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

            # If answer is not in context window, use start of context
            if offsets[context_start][0] > answer_start or offsets[context_end][1] < answer_end:
                start_positions.append(context_start)
                end_positions.append(context_start)
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
            num_proc=4
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
            fp16=True,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            dataloader_num_workers=4
        )

    def train(self):
        """Train the Longformer model on HotpotQA"""
        # Load and process dataset
        dataset = self.load_and_process_dataset()

        # Calculate training steps
        num_training_steps = (
            len(dataset["train"])
            * self.num_epochs
            // (self.batch_size * 4)
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

        # Save final model
        trainer.save_model()
        logger.info(f"Model saved to {training_args.output_dir}")

        # Final evaluation
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")
        return eval_results


def main():
    # Initialize trainer with Longformer
    trainer = LongformerHotpotTrainer(
        model_name="allenai/longformer-base-4096",
        batch_size=2,  # Smaller batch size due to longer sequences
        max_length=4096,
        num_epochs=3,
        learning_rate=2e-5,
        output_dir="./longformer_hotpot_models"
    )

    # Train the model
    trainer.train()


if __name__ == "__main__":
    main()
