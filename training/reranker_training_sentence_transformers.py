import torch
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.losses import MultipleNegativesRankingLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder import CrossEncoderTrainingArguments
import datasets
from sentence_transformers.training_args import BatchSamplers
import os
import wandb
import gc
import json

os.environ["WANDB_LOG_MODEL"] = "false"
learning_rate = 4e-6

model_path = "BAAI/bge-reranker-v2-m3"
train_dataset_path = "/Data/split/train"
valid_dataset_path = "Data/split/valid"
out_dir = f"/bge-reranker-v2-m3_lr_{learning_rate}"

train_data = datasets.load_from_disk(train_dataset_path)
valid_data = datasets.load_from_disk(valid_dataset_path)
train_data = train_data.rename_columns({'claim':'anchor', 'abstract':'positive'}).remove_columns(['claim_id','abstract_id','annotation'])
valid_data = valid_data.rename_columns({'claim':'anchor', 'abstract':'positive'}).remove_columns(['claim_id','abstract_id','annotation'])

# train_list_of_dicts = [{'sentence_1':example['claim'], 'sentence_2':example['abstract'], 'label':0} if example['annotation'] == "Not Enough Information" else {'sentence_1':example['claim'], 'sentence_2':example['abstract'], 'label':1} for example in train_data]
# # train_list_of_dicts = train_list_of_dicts + syn_data_final
# train_dict_of_lists = {
#     'sentence_1': [example['sentence_1'] for example in train_list_of_dicts],
#     'sentence_2': [example['sentence_2'] for example in train_list_of_dicts],
#     'label': [example['label'] for example in train_list_of_dicts]
# }
# train_data = datasets.Dataset.from_dict(train_dict_of_lists)

# valid_list_of_dicts = [{'sentence_1':example['claim'], 'sentence_2':example['abstract'], 'label':0} if example['annotation'] == "Not Enough Information" else {'sentence_1':example['claim'], 'sentence_2':example['abstract'], 'label':1} for example in valid_data]
# valid_dict_of_lists = {
#     'sentence_1': [example['sentence_1'] for example in valid_list_of_dicts],
#     'sentence_2': [example['sentence_2'] for example in valid_list_of_dicts],
#     'label': [example['label'] for example in valid_list_of_dicts]
# }
# valid_data = datasets.Dataset.from_dict(valid_dict_of_lists)
# print(train_data)
# print(valid_data)

model = CrossEncoder(model_path, max_length=512, tokenizer_kwargs={'model_max_length':512})

# train_loss = BinaryCrossEntropyLoss(model)
train_loss = MultipleNegativesRankingLoss(model)

training_args = CrossEncoderTrainingArguments(
    output_dir=out_dir,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    learning_rate=learning_rate,
    warmup_ratio=0.1,
    fp16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=28,
    save_strategy="steps",
    save_total_limit=1,
    logging_steps=28,
    save_steps=300,
    report_to="wandb"
)

trainer = CrossEncoderTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=valid_data,
    loss=train_loss,
)

trainer.train()

model.save(out_dir)
