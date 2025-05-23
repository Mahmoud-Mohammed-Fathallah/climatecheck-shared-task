import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import sys
import os


def calc_embeddings(bi_encoder_model, out_file_name, abstracts_df):
    # loading models
    print("loading Model...")
    retriever_zero = SentenceTransformer(bi_encoder_model, trust_remote_code=True)
    print("Model loaded successfully.")
    # calculating embeddings for the abstracts
    print("calculating embeddings for the abstracts...")
    abstracts_embeddings = retriever_zero.encode(abstracts_df["abstract"].tolist(), batch_size=4, convert_to_tensor=True, show_progress_bar=True)
    print("saving embeddings...")
    np.save(os.path.join(embedding_path, out_file_name), abstracts_embeddings.cpu().numpy())
    print(f"Abstracts embeddings shape: {abstracts_embeddings.shape}")
    # testing the loading of embeddings
    print("loading embeddings as a sanity check...")
    loaded = np.load(os.path.join(embedding_path, out_file_name))
    print(f"Abstracts embeddings shape after loading: {loaded.shape}")
    print("check succeeded!")
    return abstracts_embeddings

if __name__ == '__main__':
    if len(sys.argv) <  3:
        print("Usage: python script.py modelpath out_file_name")
        sys.exit(1)
    bi_encoder_model = sys.argv[1]
    out_file_name = sys.argv[2]
    abstracts_parquet_path = "Data/climatecheck_abstracts.parquet"
    embedding_path = "Data/embeddings"
    abstracts_df = pd.read_parquet(abstracts_parquet_path)
    calc_embeddings(bi_encoder_model, out_file_name, abstracts_df)