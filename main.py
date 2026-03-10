from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Depends
from fastapi.security import APIKeyHeader
import pandas as pd
from engine import GovernmentDataEngine
import asyncio

app = FastAPI()
engine = GovernmentDataEngine()

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key")
api_keys = {"client1": "key1", "client2": "key2"}  # placeholder API keys for clients

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key not in api_keys.values():
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# In-memory storage for client data
client_data = {}

async def run_training_pipeline(client_id: str):
    if client_id not in client_data:
        return
    sales_data = client_data[client_id]
    await asyncio.get_event_loop().run_in_executor(None, engine.train_for_client, client_id, sales_data)

@app.post("/upload-data/{client_id}")
async def upload_sales_data(client_id: str, file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    df = pd.read_csv(file.file)
    client_data[client_id] = df
    return {"status": "Data uploaded for client_id: {}".format(client_id)}

@app.post("/train/{client_id}")
async def train_model(client_id: str, background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)):
    if client_id not in client_data:
        raise HTTPException(status_code=404, detail="Client data not found")
    background_tasks.add_task(run_training_pipeline, client_id)
    return {"status": "Training started for client_id: {}".format(client_id)}

@app.get("/recommend/{client_id}")
async def get_recommendations(client_id: str, api_key: str = Depends(get_api_key)):
    # Fetch future events
    future_events = engine.fetch_gov_events()  # Assuming this fetches future events
    recommendations = engine.predict_recommendation(client_id, future_events)
    if recommendations is None:
        raise HTTPException(status_code=404, detail="Model not trained")
    return {"client_id": client_id, "recommendations": recommendations}