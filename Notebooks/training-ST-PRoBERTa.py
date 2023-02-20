from transformers import RobertaTokenizerFast, RobertaForMaskedLM, RobertaConfig, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_from_disk
import torch

max_length = 1024
vocab_size = 35
model_dir = 'Models/ST-PRoBERTa'

tokenized_train_dataset = load_from_disk('Datasets/tokenized-train-ST-PRoBERTa')
tokenized_validation_dataset = load_from_disk('Datasets/tokenized-validation-ST-PRoBERTa')
tokenized_train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'text'])
tokenized_validation_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'text'])

config = RobertaConfig(
    vocab_size=vocab_size,
    max_position_embeddings=max_length + 2,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
)

tokenizer = RobertaTokenizerFast.from_pretrained(f"{model_dir}/Tokenizer", max_len=max_length)
model = RobertaForMaskedLM(config=config)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=f"{model_dir}/Checkpoints",
    overwrite_output_dir=False,
    num_train_epochs=200,
    per_device_train_batch_size=10,
    eval_steps=10_000,
    evaluation_strategy="steps",
    metric_for_best_model="loss",
    save_steps=10_000,
    save_total_limit=100,
    prediction_loss_only=True,
    load_best_model_at_end=True,
    fp16=True,
    logging_dir=f"{model_dir}/Logs",
    ignore_data_skip=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset['train'],
    eval_dataset=tokenized_validation_dataset['train'],
)

#toggle resume_from_checkpoint
trainer.train(resume_from_checkpoint=True)
