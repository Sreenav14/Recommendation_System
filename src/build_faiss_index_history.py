import os
import numpy as np
import faiss

FAISS_DIR = "../artifact/tfrs_retrival_model_history/faiss"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
EMB_PATH = os.path.join(FAISS_DIR, "items_embeddings.npy")

def main():
    os.makedirs(FAISS_DIR, exist_ok=True)

    embeddings = np.load(EMB_PATH).astype("float32")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    print("FAISS index built")
    print("Total vectors indexed:", index.ntotal)
    print("saved index to", INDEX_PATH)

if __name__ == "__main__":
    main()
