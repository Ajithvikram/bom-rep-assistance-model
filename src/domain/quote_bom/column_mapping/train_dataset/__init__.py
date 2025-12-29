import numpy as np
import faiss
from src.domain.quote_bom.column_mapping.extract_features import getEmbeddings
import random
from tensorflow.keras.models import load_model

def buildTrainingPairs():
    print("Training pairs started...")
    dataset = [
        ("Qty 1", "Quantity 1"),
        ("Volume 1", "Quantity 1"),
        ("Qty 2", "Quantity 2"),
        ("Volume 2", "Quantity 2"),
        ("Qty 3", "Quantity 3"),
        ("Volume 3", "Quantity 3"),
    ]
    positives = [(a, c, 1) for a, c in dataset]

    # Negative pairs
    canonicals = list(sorted(set([c for _, c in dataset])))
    negatives = []
    for alias, canonical in dataset:
        wrong = random.choice([c for c in canonicals if c != canonical])
        negatives.append((alias, wrong, 0))
    print("Training pairs ended...")
    return positives + negatives

def getTrainedSet():
    try:
        print("Training set started...")
        
        pairs = buildTrainingPairs()
        listX1 = [a for a, c, label in pairs]
        listX2 = [c for a, c, label in pairs]
        y = [label for a, c, label in pairs]

        X1 = np.array(getEmbeddings(listX1))
        X2 = np.array(getEmbeddings(listX2))
        y = np.array(y)
        print("shape - " + str(X1.shape[0]) + " , " + str(X1.shape[1]))
        embeddingDim = X1.shape[1]

        print("Training set ended...")
        return X1, X2, y, embeddingDim
    except Exception as e:
        print("Error occurred:", e)

def getIndexes ():
    dataset = [
        ("Qty 1", "Quantity 1"),
        ("Volume 1", "Quantity 1"),
        ("Qty 2", "Quantity 2"),
        ("Volume 2", "Quantity 2"),
        ("Qty 3", "Quantity 3"),
        ("Volume 3", "Quantity 3"),
    ]
    canonical_names = sorted(set([c for _, c in dataset]))
    model = load_model("models/alias.keras")
    canonical_embs = model.predict(getEmbeddings(canonical_names))
    dim = canonical_embs.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(canonical_embs).astype("float32"))
    
def findAliasis(alias):
    vec = getEmbeddings([alias])
    model = load_model("models/alias.keras")
    vec = model.predict(vec).astype("float32")
    index = getIndexes()
    
    D, I = index.search(vec, k=1)  # find nearest canonical
    best_index = I[0][0]
    dataset = [
        ("Qty 1", "Quantity 1"),
        ("Volume 1", "Quantity 1"),
        ("Qty 2", "Quantity 2"),
        ("Volume 2", "Quantity 2"),
        ("Qty 3", "Quantity 3"),
        ("Volume 3", "Quantity 3"),
    ]
    canonical_names = sorted(set([c for _, c in dataset]))
    
    return canonical_names[best_index], float(D[0][0])
