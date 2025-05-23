#!/usr/bin/env python
import sys
import os
import os.path
import ast
import pandas as pd
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

######### Subtask I - Retrieval Evaluation ########

def recall_at_k(gold_data, predictions, k=5):

    # Keep only relevant documents
    relevant_labels = {"Supports", "Refutes"}
    gold_data = gold_data[gold_data['annotation'].isin(relevant_labels)]

    recall_scores = []
    golden_df = gold_data.groupby("claim_id", as_index=False).agg(lambda x: list(x))
    retrieved_df = predictions.groupby("claim_id", as_index=False).agg(lambda x: list(x))

    for claim_id in golden_df['claim_id'].tolist():
        golden_abstracts = golden_df[golden_df.claim_id == claim_id]['abstract_original_index'].tolist()[0]
        retrieved_abstracts = retrieved_df[retrieved_df.claim_id == claim_id].sort_values(by='rank')['abstract_id'].tolist()
        if len(retrieved_abstracts) >= 1:
          retrieved_abstracts = retrieved_abstracts[0]

        if k > len(retrieved_abstracts):
          recall = sum(1 for t in golden_abstracts if t in retrieved_abstracts) / len(golden_abstracts)
        else:
          recall = sum(1 for t in golden_abstracts if t in retrieved_abstracts[:k]) / len(golden_abstracts)


        recall_scores.append(recall)

    recall_at_k_score = sum(recall_scores) / len(recall_scores) if recall_scores else 0

    if(np.isnan(recall_at_k_score)):
        recall_at_k_score = 404

    return recall_at_k_score

def bpref(gold_data, predictions):

    relevant_labels = {"Supports", "Refutes"}
    non_relevant_label = "Not Enough Information"

    bpref_scores = []

    for claim_id in predictions['claim_id'].unique():
        # Get relevant and non-relevant sets from gold data
        gold_subset = gold_data[gold_data['claim_id'] == claim_id]
        relevant_docs = set(gold_subset[gold_subset['annotation'].isin(relevant_labels)]['abstract_original_index'])
        non_relevant_docs = set(gold_subset[gold_subset['annotation'] == non_relevant_label]['abstract_original_index'])

        if not relevant_docs or not non_relevant_docs:
            continue  # Skip if no relevant or non-relevant docs

        retrieved_docs = predictions[predictions['claim_id'] == claim_id].sort_values(by='rank')
        R = len(relevant_docs)

        irrelevant_count = 0
        bpref_sum = 0
        relevant_seen = 0

        for _, row in retrieved_docs.iterrows():
            doc_id = row['abstract_id']

            if doc_id in relevant_docs:
                bpref_sum += 1 - (irrelevant_count / R)
                relevant_seen += 1
            elif doc_id in non_relevant_docs:
                if irrelevant_count < R:
                    irrelevant_count += 1

            if relevant_seen == R:
                break

        if R > 0:
            bpref_scores.append(bpref_sum / R)

    bpref_score = sum(bpref_scores) / len(bpref_scores) if bpref_scores else 0.0

    return bpref_score

######### Subtask II - Classification Evaluation ########

def claim_verification_scores(gold_data, predictions):

    # Check if the predictions file contains a "label" column --> If not, that means the team is not participating in subtask II
    if 'label' not in predictions.columns:
        return 404, 404, 404

    # Step 1: iterate over the predictions and check if the claim-abstract pair exists in the gold data
    # create a new list of predictions that consists of claim-abstract pairs only if they exist in the gold data
    predicted_labels = []
    gold_labels = []
    C_correct = 0  # Counter for correctly predicted claim-abstract pairs

    for idx, row in predictions.iterrows():
        claim_id = row['claim_id']
        abstract_id = row['abstract_id']

        if gold_data[(gold_data['claim_id'] == claim_id) & (gold_data['abstract_original_index'] == abstract_id)].shape[0] > 0:
            
            gold_label = gold_data[(gold_data['claim_id'] == claim_id) & (gold_data['abstract_original_index'] == abstract_id)]['annotation'].values[0].lower()
            pred_label = row['label'].lower()
            gold_labels.append(gold_label)
            predicted_labels.append(pred_label)

            if pred_label == gold_label:
                C_correct += 1


    assert len(gold_labels) == len(predicted_labels), "The number of gold labels and predicted labels does not match."

    # Step 2: calculate precision, recall, and F1
    precision = precision_score(gold_labels, predicted_labels, average='weighted')
    if(np.isnan(precision)):
        precision = 404

    recall = recall_score(gold_labels, predicted_labels, average='weighted')
    if(np.isnan(recall)):
        recall = 404

    f1 = f1_score(gold_labels, predicted_labels, average='weighted')
    if(np.isnan(f1)):
        f1 = 404

    return precision, recall, f1


######### Main Script #########

if len(sys.argv) != 4:
    print("Usage: python evaluate.py <input_path> <output_path> <ground_truth_path>")
    sys.exit(1)
input_path = sys.argv[1]
output_path = sys.argv[2]
gt_path = sys.argv[3]

# submit_dir = os.path.join(input_dir, 'res')
# truth_dir = os.path.join(input_dir, 'ref')

if __name__ == "__main__":
    output_file = open(output_path, 'w')

    gold_data = pd.read_csv(gt_path)
    predictions = pd.read_csv(input_path)

    ###### Scores for Subatask I ###### 
    recall_2_score = recall_at_k(gold_data, predictions, k=2)
    recall_5_score = recall_at_k(gold_data, predictions, k=5)
    recall_10_score = recall_at_k(gold_data, predictions, k=10)
    recall_100_score = recall_at_k(gold_data, predictions, k=100)
    recall_500_score = recall_at_k(gold_data, predictions, k=500)
    recall_1000_score = recall_at_k(gold_data, predictions, k=1000)
    bpref_score = bpref(gold_data, predictions)
    all_subtaskI_scores = [recall_2_score, recall_5_score, recall_10_score, bpref_score]
    
    if 404 in all_subtaskI_scores:
        subtaskI_score = 404
    else:
        subtaskI_score = np.mean(all_subtaskI_scores)

    ###### Scores for Subatask II ######
    precision, recall, f1 = claim_verification_scores(gold_data, predictions)
    # precision, recall, f1 = 404,404,404
    if 404 in [precision, recall, f1]:
        subtaskII_score = 404
    else:
        subtaskII_score = f1 + recall_10_score


    output_file.write("Recall@2: "+str(recall_2_score)+"\n")
    output_file.write("Recall@5: "+str(recall_5_score)+"\n")
    output_file.write("Recall@10: "+str(recall_10_score)+"\n")
    output_file.write("Recall@100: "+str(recall_100_score)+"\n")
    output_file.write("Recall@500: "+str(recall_500_score)+"\n")
    output_file.write("Recall@1000: "+str(recall_1000_score)+"\n")
    output_file.write("B-pref: "+str(bpref_score)+"\n")
    output_file.write("SubtaskI-Score: "+str(subtaskI_score)+"\n")
    output_file.write("Precision: "+str(precision)+"\n")
    output_file.write("Recall: "+str(recall)+"\n")
    output_file.write("F1: "+str(f1)+"\n")
    output_file.write("SubtaskII-Score: "+str(subtaskII_score)+"\n")

    output_file.close()
