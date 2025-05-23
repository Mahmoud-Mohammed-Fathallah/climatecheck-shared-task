import torch
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers import SentenceTransformerTrainingArguments
import datasets
import os
import wandb
import gc
import json
os.environ["WANDB_LOG_MODEL"] = "false"
learning_rate = 1e-5
print(f"Training with lr={learning_rate}")
model_path = "NovaSearch/stella_en_400M_v5"
train_dataset_path = "Data/good_split/train"
valid_dataset_path = "Data/good_split/valid"
out_dir = f"stella_en_400M_v5_lr_{learning_rate}"

train_data = datasets.load_from_disk(train_dataset_path)
valid_data = datasets.load_from_disk(valid_dataset_path)
train_data = train_data.rename_columns({'claim':'anchor', 'abstract':'positive'}).remove_columns(['claim_id','abstract_id','annotation'])
valid_data = valid_data.rename_columns({'claim':'anchor', 'abstract':'positive'}).remove_columns(['claim_id','abstract_id','annotation'])
# with open(syn_data_path, 'r') as f:
#     syn_data = json.loads(f.read())
# syn_data_final = []
# supp_count = 0
# ref_count = 0
# not_count = 0
# for element in syn_data:
#     to_add = {}
#     to_add['sentence_1'] = element['claim']
#     to_add['sentence_2'] = element['abstract']
#     if element['annotation'].lower() == 'supports':
#         to_add['label'] = 1
#         supp_count += 1
#     elif element['annotation'].lower() == 'refutes':
#         to_add['label'] = 1
#         ref_count += 1
#     else:
#         to_add['label'] = 0
#         not_count += 1

#     syn_data_final.append(to_add)

# print(f"support count: {supp_count}, refute count: {ref_count}, not enough info count: {not_count}")
# # assert supp_count == not_count, "problem with syn data"

# train_list_of_dicts = [{'sentence_1':example['claim'], 'sentence_2':example['abstract'], 'label':0} if example['annotation'] == "Not Enough Information" else {'sentence_1':example['claim'], 'sentence_2':example['abstract'], 'label':1} for example in train_data]
# train_list_of_dicts = train_list_of_dicts + syn_data_final
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
print(train_data)
print(valid_data)


model = SentenceTransformer(model_path, trust_remote_code=True, tokenizer_kwargs={'model_max_length':512})

# train_loss = losses.ContrastiveLoss(model)
train_loss = losses.MultipleNegativesRankingLoss(model)

training_args = SentenceTransformerTrainingArguments(
    output_dir=out_dir,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    learning_rate=learning_rate,
    warmup_ratio=0.1,
    logging_dir='./logs',
    logging_steps=20,
    save_steps=1000,
    save_total_limit=3,
    evaluation_strategy='steps',
    eval_steps=20,
    report_to="wandb"
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
