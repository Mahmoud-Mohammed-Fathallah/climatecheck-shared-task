import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import multiprocessing
import numpy as np
import sys
import os

def split_data(data, num_chunks):
    k, m = divmod(len(data), num_chunks)
    return [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_chunks)]

def compute_embeddings(device_id, sentences, model_name, return_dict):
    device = f"cuda:{device_id}"
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    model.eval()
    embeddings = model.encode(sentences, batch_size=4, convert_to_tensor=True, show_progress_bar=True)
    return_dict[device_id] = embeddings.cpu().numpy()

if __name__ == '__main__':
    if len(sys.argv) <  3:
        print("Usage: python script.py modelpath out_file_name")
        sys.exit(1)
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    bi_encoder_model = sys.argv[1]
    out_file_name = sys.argv[2]
    abstracts_parquet_path = "Data/climatecheck_abstracts.parquet"
    embedding_path = "Data/embeddings"
    abstracts_df = pd.read_parquet(abstracts_parquet_path)
    
    documents = abstracts_df["abstract"].tolist()
    document_chunks = split_data(documents, num_gpus)
    multiprocessing.set_start_method('spawn', force=True)  # Necessary for CUDA with multiprocessing
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    
    for i in range(num_gpus):
        p = multiprocessing.Process(target=compute_embeddings, args=(i, document_chunks[i], bi_encoder_model, return_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        
    
    all_embeddings = []
    for i in range(num_gpus):
        all_embeddings.extend(return_dict[i])
    
    print("saving embeddings...")
    np.save(os.path.join(embedding_path, out_file_name), np.array(all_embeddings))
    print(f"Abstracts embeddings shape: {all_embeddings.shape}")