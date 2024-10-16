from transformers import T5TokenizerFast, T5ForQuestionAnswering
from squad_pp import *

# Load tokenizer and model
model_name = "t5-base"
tokenizer = T5TokenizerFast.from_pretrained(model_name)
model = T5ForQuestionAnswering.from_pretrained(model_name)

# Load and prepare dataset
squad_dataset = load_squad_dataset(version="2.0")  # or "1.1" for SQuAD v1.1
tokenized_dataset = prepare_dataset(squad_dataset, tokenizer)

# Configure LoRA
lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.QUESTION_ANS
)

# Wrap model with LoRA
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./t5_lora_squad",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)
