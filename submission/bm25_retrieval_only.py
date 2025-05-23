import bm25s
import pandas as pd
import datasets
import os
from tqdm import tqdm
import json

abstracts_parquet_path = "Data/climatecheck_abstracts.parquet"
valid_dataset_path = "Data/split/valid"
out_dir = "./retriever_only"
out_file_name = "bm25_10.csv"

abstracts_df = pd.read_parquet(abstracts_parquet_path)
climate_dataset_valid = datasets.load_from_disk(valid_dataset_path)

corpus = abstracts_df['abstract'].tolist()
corpus_tokens = bm25s.tokenize(corpus)
retriever = bm25s.BM25(corpus=corpus)
retriever.index(corpus_tokens)

visited = set()
submission_dict = {
    "claim_id": [],
    "abstract_id": [],
    "rank": []
}
for element in tqdm(climate_dataset_valid):
    if element['claim_id'] not in visited:
        visited.add(element['claim_id'])
        query_tokens = bm25s.tokenize(element['claim'])
        top_docs, scores = retriever.retrieve(query_tokens, k=10)
        submission_dict['claim_id'].extend([element['claim_id']] * len(top_docs[0]))
        submission_dict['abstract_id'].extend([abstracts_df.loc[abstracts_df['abstract'] == doc, 'abstract_id'].iloc[0]for doc in top_docs[0]])
        submission_dict['rank'].extend(list(range(len(top_docs[0]))))
submission_df = pd.DataFrame(submission_dict)
submission_df.to_csv(os.path.join(out_dir, out_file_name), index=False)