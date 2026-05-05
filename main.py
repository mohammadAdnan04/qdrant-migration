from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoModel
import os

app = FastAPI(title="Jina Reranker V3 API", version="1.0")

# Determine device (Coolify might just be CPU unless you've specifically configured GPU pass-through)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...")

# Load model globally when the app starts
model = AutoModel.from_pretrained(
    "jinaai/jina-reranker-v3", 
    trust_remote_code=True,
)
model.to(device)
model.eval()
print("Model loaded successfully!")

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_n: int = 10

@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "healthy", "device": str(device)}

@app.post("/rerank")
async def rerank_endpoint(request: RerankRequest):
    try:
        with torch.no_grad():
            # Jina's native rerank method
            results = model.rerank(
                request.query, 
                request.documents, 
                top_n=request.top_n
            )
            
            # Convert numpy types to native Python types so FastAPI can JSON serialize them
            clean_results = []
            for r in results:
                clean_r = {}
                for k, v in r.items():
                    if hasattr(v, 'item'):  # catches numpy scalars like numpy.float32
                        clean_r[k] = v.item()
                    else:
                        clean_r[k] = v
                clean_results.append(clean_r)
                
        return {"results": clean_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
