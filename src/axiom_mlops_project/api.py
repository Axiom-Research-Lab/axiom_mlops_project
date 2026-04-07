from fastapi import FastAPI
from pydantic import BaseModel, Field
import time

app = FastAPI(title="Axiom Instant EFT Engine", version="1.0.0")


class EFTRequest(BaseModel):
    fund_amount: float = Field(..., example=1500.0)
    fund_type_encoded: int = Field(..., example=1)  # 1 for Instant EFT
    hour_of_day: int = Field(..., gt=-1, lt=24, example=14)


@app.post("/v1/predict")
async def predict_eft(request: EFTRequest):
    start_time = time.time()
    # High-level demo logic: Large amounts at odd hours are flagged
    is_risky = (request.fund_amount > 12000) or (request.hour_of_day < 5)
    prob = 0.98 if not is_risky else 0.35

    return {
        "status": "SUCCESS" if prob > 0.5 else "DECLINED",
        "confidence": prob,
        "metadata": {
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "model_version": "2026.04.07-v1",
        },
    }


@app.get("/health")
async def health():
    return {"status": "online", "system": "Axiom-VodaPay"}
