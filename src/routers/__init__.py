from fastapi import APIRouter
from src.domain.quote_bom.column_mapping.train_model import trainModel

router = APIRouter(prefix="/quote-bom")

@router.get("/train")
def train_bom_model():
    # value = findAlias("Qty 1")
    trainModel()
    return {"message": "Training BOM model..."}
