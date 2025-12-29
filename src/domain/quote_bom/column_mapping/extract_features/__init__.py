import numpy as np
from sentence_transformers import SentenceTransformer

# Embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def getEmbeddings(names):
    return embed_model.encode(names, batch_size=64, normalize_embeddings=True)
