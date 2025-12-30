import os
import numpy as np
import faiss
import json

EMB_DIR = "../artifact/tfrs_retrival_model_hardneg/faiss"
INDEX_DIR = "../artifact/tfrs_retrival_model_hardneg/faiss"
os.makedirs(INDEX_DIR, exist_ok=True)

def main():
    
    # load embeddings
    embeddings = np.load(os.path.join(EMB_DIR, "items_embeddings.npy")).astype("float32")
    
    # Embedding dimension
    dim = embeddings.shape[1]
    
    # use inner product 
    index = faiss.IndexFlatIP(dim)
    
    # add all movie vectors
    index.add(embeddings)
    
    # save index
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
    
    print("FAISS index built")
    print("Total vectors indexed:", index.ntotal)
    print("saved index to", os.path.join(INDEX_DIR, "index.faiss"))
    
if __name__ == "__main__":
    main()