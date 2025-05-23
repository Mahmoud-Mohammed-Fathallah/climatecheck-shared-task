import datasets
import random
import matplotlib.pyplot as plt
from collections import Counter

dataset_path = "rabuahmad/climatecheck"

dataset = datasets.load_dataset(dataset_path)
train_dataset = dataset["train"]
print(dataset)
claim_ids_unique = set([e['claim_id'] for e in train_dataset])
print(f"there are {len(train_dataset)} ids, and only {len(claim_ids_unique)} are unique")
random.seed(22)
random_sample = random.sample(list(claim_ids_unique), 50)
print(random_sample)
# create a new dataset with only the random sample
valid = {
    "claim_id": [],
    "abstract_id": [],
    "claim": [],
    "abstract": [],
    "annotation": []
}
train = {
    "claim_id": [],
    "abstract_id": [],
    "claim": [],
    "abstract": [],
    "annotation": []
}
for element in train_dataset:
    if element['claim_id'] in random_sample:
        valid["claim_id"].append(element["claim_id"])
        valid["abstract_id"].append(element["abstract_id"])
        valid["claim"].append(element["claim"])
        valid["abstract"].append(element["abstract"])
        valid["annotation"].append(element["annotation"])
    else:
        train["claim_id"].append(element["claim_id"])
        train["abstract_id"].append(element["abstract_id"])
        train["claim"].append(element["claim"])
        train["abstract"].append(element["abstract"])
        train["annotation"].append(element["annotation"])

# print(train)
# print(valid)
train = datasets.Dataset.from_dict(train)
valid = datasets.Dataset.from_dict(valid)
print(train)
print(valid)
exit(0)

# save the new dataset to a file
train.save_to_disk("Data/split/train")
valid.save_to_disk("Data/split/valid")
# # load the new dataset from disk
train_dataset = datasets.load_from_disk("Data/split/train")
valid_dataset = datasets.load_from_disk("Data/split/valid")
print(train_dataset)
print(valid_dataset)
counter_train = Counter(train_dataset["annotation"])
counter_valid = Counter(valid_dataset["annotation"])
print(counter_train)
print(counter_valid)

plt.bar(counter_train.keys(), counter_train.values())
plt.title("Train Dataset Annotations")
plt.xlabel("Annotation")
plt.ylabel("Count")
plt.savefig("/Data/train_dataset_annotations.png")
plt.clf()
plt.bar(counter_valid.keys(), counter_valid.values())
plt.title("Valid Dataset Annotations")
plt.xlabel("Annotation")
plt.ylabel("Count")
plt.savefig("/Data/valid_dataset_annotations.png")
