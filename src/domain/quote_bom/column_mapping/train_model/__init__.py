from src.domain.quote_bom.column_mapping.create_model import createModel 
from src.domain.quote_bom.column_mapping.train_dataset import getTrainedSet

def trainModel():
    print("Training Started...")
    X1, X2, y, embeddingDim = getTrainedSet()
    print("Created Training Set...")
    model, tower = createModel(embeddingDim)
    print("Created Model...")

    model.fit([X1, X2], y, epochs=50, batch_size=4)
    print("Training the Model...")

    tower.save("models/alias.keras")
    print("Training Ended...")
