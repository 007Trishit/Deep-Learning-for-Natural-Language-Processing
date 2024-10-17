import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from evaluate import load

# Load SQuAD dataset


def load_squad_dataset(version="2.0"):
    name = "squad_v2" if version == "2.0" else "squad"
    dataset = load_dataset(name)
    return dataset

# Tokenize function


def tokenize_function(examples, tokenizer, max_length=384):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=128,
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
        if len(answer["answer_start"]) > 0:
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
        else:
            # No answer case
            start_char = end_char = 0
        # start_char = answer["answer_start"][0]
        # end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
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
            # Otherwise it's the start and end token positions
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

# Prepare dataset


def prepare_dataset(dataset, tokenizer):
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    return tokenized_dataset

# Evaluation function


def compute_metrics(eval_pred, version="2.0"):
    metric = load("squad_v2" if version == "2.0" else "squad")
    logits, labels = eval_pred
    start_logits, end_logits = logits
    start_positions, end_positions = labels
    predictions = [
        {"start_logits": start_logits[i], "end_logits": end_logits[i]}
        for i in range(len(start_logits))
    ]
    references = [
        {"start_position": start_positions[i],
            "end_position": end_positions[i]}
        for i in range(len(start_positions))
    ]
    return metric.compute(predictions=predictions, references=references)
