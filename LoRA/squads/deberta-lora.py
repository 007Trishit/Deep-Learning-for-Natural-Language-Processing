from transformers import DebertaV2TokenizerFast, DebertaV2ForQuestionAnswering
from squad_pp import *

# Load tokenizer and model
model_name = "microsoft/deberta-v3-base"
tokenizer = DebertaV2TokenizerFast.from_pretrained(model_name)
model = DebertaV2ForQuestionAnswering.from_pretrained(model_name)

# Load and prepare dataset
squad_dataset = load_squad_dataset()  # 1.1 default or "2.0" for SQuAD v2.0
tokenized_dataset = prepare_dataset(squad_dataset, tokenizer)

# Configure LoRA
lora_config = LoraConfig(
    r=RANK,
    lora_alpha=32,
    target_modules=["query_proj", "value_proj", "key_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.QUESTION_ANS
)

# Wrap model with LoRA
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir=f"./saved_models_hp/2.0/deberta_lora_squad_{RANK}_{targets}",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
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
    # compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)
