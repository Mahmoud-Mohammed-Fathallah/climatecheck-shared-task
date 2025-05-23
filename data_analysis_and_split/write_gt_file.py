import pandas as pd
import datasets

valid_dataset_path = "Data/split/valid"
val_dataset = datasets.load_from_disk(valid_dataset_path)
print(val_dataset)
fin_dict = {
    "claim_id": [],
    "abstract_original_index": [],
    "annotation": []
}
for element in val_dataset:
    fin_dict["claim_id"].append(element["claim_id"])
    fin_dict["abstract_original_index"].append(element["abstract_id"])
    fin_dict["annotation"].append(element["annotation"])
    
gt_df = pd.DataFrame(fin_dict)
gt_df.to_csv("./climatecheck_gt.csv", index=False)