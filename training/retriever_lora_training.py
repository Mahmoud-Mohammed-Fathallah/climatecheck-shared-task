import torch
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers import SentenceTransformerTrainingArguments
import datasets
from peft import LoraConfig, TaskType
import os

os.environ["WANDB_LOG_MODEL"] = "false"
learning_rate = 1e-6
print(f"Training with lr={learning_rate}")
model_path = "infly/inf-retriever-v1-1.5b"
train_dataset_path = "Data/good_split/train"
valid_dataset_path = "Data/good_split/valid"
out_dir = f"/inf-retriever-v1-1.5b_lr_{learning_rate}"

train_data = datasets.load_from_disk(train_dataset_path)
valid_data = datasets.load_from_disk(valid_dataset_path)
train_data = train_data.rename_columns({'claim':'anchor', 'abstract':'positive'}).remove_columns(['claim_id','abstract_id','annotation'])
valid_data = valid_data.rename_columns({'claim':'anchor', 'abstract':'positive'}).remove_columns(['claim_id','abstract_id','annotation'])
print(train_data)
print(valid_data)

lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=128,
    lora_alpha=256,
    lora_dropout=0.1,
)

model = SentenceTransformer(model_path, trust_remote_code=True, tokenizer_kwargs={'model_max_length':512}, model_kwargs={"torch_dtype": torch.float16})
model.max_seq_length = 512
model.add_adapter(lora_config)
model.train()

train_loss = losses.MultipleNegativesRankingLoss(model)

training_args = SentenceTransformerTrainingArguments(
    output_dir=out_dir,
    num_train_epochs=4,
    fp16=False,
    per_device_train_batch_size=8,
    learning_rate=learning_rate,
    warmup_ratio=0.1,
    logging_dir='./logs',
    logging_steps=8,
    save_steps=2000,
    save_total_limit=3,
    evaluation_strategy='steps',
    eval_steps=8,
    report_to="wandb",
    max_grad_norm=1.0
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=valid_data,
    loss=train_loss
)

trainer.train()

model.save(out_dir)