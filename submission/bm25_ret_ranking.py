import torch
import datasets
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from tqdm import tqdm
import numpy as np
import gc
import sys
import os
import bm25s
import json


def retrieve_top_k(retriever, query, abstracts_df, k=2):
    query_tokens = bm25s.tokenize(query)
    top_docs, scores = retriever.retrieve(query_tokens, k=k)
    ids = [abstracts_df.loc[abstracts_df['abstract'] == doc, 'abstract_id'].iloc[0] for doc in top_docs[0]]
    return ids, top_docs[0]

def rank_retrieved_docs(ranker, query, retrieved_docs, indices):
    with torch.no_grad():
        scores = ranker.predict([(query, doc) for doc in retrieved_docs])
    reranked_docs = sorted(zip(retrieved_docs, indices, scores), key=lambda x: x[2], reverse=True)
    return reranked_docs

def get_top_related_abstracts(query, retriever, ranker, abstracts_df, k_retriever=100, k_final=10):
    top_k_indices, retrieved_docs = retrieve_top_k(retriever, query, abstracts_df, k_retriever)
    reranked_docs = rank_retrieved_docs(ranker, query, retrieved_docs, top_k_indices)
    return reranked_docs[:k_final]


def main(climate_dataset_test, retriever, ranker, abstracts_df, out_dir, out_file_name):
    # inference and creating submission file
    # schema for submission csv file:
    # claim_id, abstract_id, rank
    submission_dict = {
        "claim_id": [],
        "abstract_id": [],
        "rank": []
    }
    for element in tqdm(climate_dataset_test):
        reranked_docs = get_top_related_abstracts(
            element['claim'],
            retriever,
            ranker,
            abstracts_df,
            k_retriever=1000,
            k_final=10
        )
        torch.cuda.empty_cache()
        gc.collect()
        submission_dict['claim_id'].extend([element['claim_id']] * len(reranked_docs))
        submission_dict['abstract_id'].extend([list(abstracts_df['abstract_id'])[doc[1]] for doc in reranked_docs])
        submission_dict['rank'].extend(list(range(len(reranked_docs))))
        del reranked_docs
        gc.collect()
    submission_df = pd.DataFrame(submission_dict)
    submission_df.to_csv(os.path.join(out_dir, out_file_name), index=False)

if __name__ == '__main__':
    if len(sys.argv) <  4:
        print("Usage: python script.py ranker_model_path out_file_name (valid or test)")
        sys.exit(1)
        
    cross_encoder_model = sys.argv[1]
    out_file_name = sys.argv[2]
    validation = sys.argv[3]
    
    ranker_max_len = 512
    abstracts_parquet_path = "Data/climatecheck_abstracts.parquet"
    out_dir = "./retrieval_ranking"
    dataset_path = "rabuahmad/climatecheck"
    valid_dataset_path = "Data/split/valid"

    climate_dataset = datasets.load_dataset(dataset_path)
    print(climate_dataset)
    climate_dataset_test = climate_dataset['test']
    climate_dataset_valid = datasets.load_from_disk(valid_dataset_path)
    abstracts_df = pd.read_parquet(abstracts_parquet_path)
    print(list(abstracts_df['abstract_id'])[0])
    
    # loading models
    print("loading Models...")
    corpus = abstracts_df['abstract'].tolist()
    # Initialize BM25
    corpus_tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(corpus_tokens)
    
    ranker = CrossEncoder(cross_encoder_model, max_length=ranker_max_len, tokenizer_kwargs={'model_max_length':ranker_max_len})
    ranker.eval()
    print("Models loaded successfully.")

    # running the main function for submission
    if validation != 'test':
        unique_elements = []
        visited_ids = set()
        for element in climate_dataset_valid:
            if element['claim_id'] not in visited_ids:
                unique_elements.append(element)
                visited_ids.add(element['claim_id'])
        main(unique_elements, retriever, ranker, abstracts_df, out_dir, out_file_name)
    else:
        main(climate_dataset_test, retriever, ranker, abstracts_df, out_dir, out_file_name)
    print("Submission file created successfully.")
    print(f"Submission file saved at: {os.path.join(out_dir, out_file_name)}")