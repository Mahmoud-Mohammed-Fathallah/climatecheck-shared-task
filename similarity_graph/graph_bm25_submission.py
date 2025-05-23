import torch
import datasets
import pandas as pd
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import numpy as np
import gc
import sys
import os
import bm25s
import json


class SimGraph():
    def __init__(self, edges_path, weights_path, docs):
        self.edges = np.load(edges_path)
        self.weights = np.load(weights_path)
        self.docnos = [doc['docno'] for doc in docs]
        self.texts = [doc['text'] for doc in docs]
        self.docno_to_idx = {docno: idx for idx, docno in enumerate(self.docnos)}
        self.idx_to_docno = {idx: docno for docno, idx in self.docno_to_idx.items()}

    def get_neighbours(self, document_no):
        idx = self.docno_to_idx[document_no]
        neighbours = self.edges[idx]
        scores = self.weights[idx]
        text_to_return = [self.texts[idx] for idx in neighbours]
        return neighbours, scores, text_to_return

def retrieve_top_k(retriever, query, abstracts_df, k=2):
    query_tokens = bm25s.tokenize(query)
    top_docs, scores = retriever.retrieve(query_tokens, k=k)
    ids = [abstracts_df.loc[abstracts_df['abstract'] == doc, 'abstract_id'].iloc[0] for doc in top_docs[0]]
    return ids, top_docs[0]

def iterative_ranking(ranker, query, initial_retrieved_docs, indices, graph, k):
    retrieved_docs = initial_retrieved_docs
    for i in range(20):
        print(f"iteration {i+1}")
        with torch.no_grad():
            scores = ranker.predict([(query, doc) for doc in retrieved_docs])
        reranked_docs = sorted(zip(retrieved_docs, indices, scores), key=lambda x: x[2], reverse=True)
        top_k_idx = [t[1] for t in reranked_docs[:k]]
        retrieved_docs = [t[0] for t in reranked_docs[:k]]
        indices = [t[1] for t in reranked_docs[:k]]
        for t in top_k_idx:
            neighbours, scores, text_to_return = graph.get_neighbours(t)
            for n,t in zip(neighbours, text_to_return):
                if n not in indices:
                    retrieved_docs.append(t)
                    indices.append(n)
            assert len(retrieved_docs) == len(indices)
    return reranked_docs

def get_top_related_abstracts(query, retriever, ranker, abstracts_df, graph, k_retriever=1000, k_intermediate=100, k_final=10):
    top_k_indices, retrieved_docs = retrieve_top_k(retriever, query, abstracts_df, k_retriever)
    reranked_docs = iterative_ranking(ranker, query, retrieved_docs, top_k_indices, graph, k=k_intermediate)
    return reranked_docs[:k_final]


def main(climate_dataset_test, retriever, ranker, abstracts_df, out_dir, out_file_name, graph):
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
            graph,
            k_retriever=50,
            k_final=10,
            k_intermediate=30
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
    
def read_augmented_queries(file_path):
    with open(file_path, 'r') as f:
        queries_list = json.loads(f.read())
    queries_to_return = []
    for query in queries_list:
        queries_to_return.append(
            {
                'claim': query['claim'] + " " + query['keywords'],
                'claim_id': query['claim_id']
            }
        )
    return queries_to_return

if __name__ == '__main__':
    if len(sys.argv) <  4:
        print("Usage: python script.py ranker_model_path out_file_name (valid or test)")
        sys.exit(1)
        
    cross_encoder_model = sys.argv[1]
    out_file_name = sys.argv[2]
    validation = sys.argv[3]
    
    ranker_max_len = 512
    abstracts_parquet_path = "Data/climatecheck_abstracts.parquet"
    out_dir = "code/similarity_graph"
    dataset_path = "rabuahmad/climatecheck"
    valid_dataset_path = "Data/split/valid"
    edges1 = "code/similarity_graph/sim_graph/edges.npy"
    weights1 = "code/similarity_graph/sim_graph/weights.npy"

    climate_dataset = datasets.load_dataset(dataset_path)
    print(climate_dataset)
    climate_dataset_test = climate_dataset['test']
    climate_dataset_valid = datasets.load_from_disk(valid_dataset_path)
    abstracts_df = pd.read_parquet(abstracts_parquet_path)
    print(list(abstracts_df['abstract_id'])[0])

    corpus = abstracts_df['abstract'].tolist()
    ids = abstracts_df['abstract_id'].tolist()
    docs = []
    print("preparing data")
    for abstract, abs_id in tqdm(zip(corpus, ids)):
        obj = {}
        obj['text'] = abstract
        obj['docno'] = abs_id
        docs.append(obj)

    graph = SimGraph(edges1, weights1, docs)
    
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
        main(unique_elements, retriever, ranker, abstracts_df, out_dir, out_file_name, graph)
    else:
        main(climate_dataset_test, retriever, ranker, abstracts_df, out_dir, out_file_name, graph)
    print("Submission file created successfully.")
    print(f"Submission file saved at: {os.path.join(out_dir, out_file_name)}")