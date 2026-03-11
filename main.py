from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
import pandas as pd
from engine import GovernmentDataEngine
import asyncio

app = FastAPI(title="Food and Beverages SME Recommendation API", description="Free API for product recommendations during events using Machine Learning.")
engine = GovernmentDataEngine()

@app.post("/upload-data/")
async def upload_sales_data(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    return {"status": "Data uploaded successfully"}

@app.post("/train/")
async def train_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training_pipeline, 'default')
    return {"status": "Training started for default client"}

async def run_training_pipeline(file: UploadFile = File(...)):
    sales_df = pd.read_csv(file.file)
    if sales_df is not None:
        await asyncio.get_event_loop().run_in_executor(None, engine.train_for_client, 'default', sales_df)

@app.post("/test/")
async def test_model(file: UploadFile = File(...)):
    test_df = pd.read_csv(file.file)
    results = engine.test_model('default', test_df)
    if results is None:
        raise HTTPException(status_code=404, detail="Model not trained")
    return {"client_id": 'default', "test_results": results}

@app.get("/recommend/")
async def get_recommendations():
    """Return a list of top products predicted for upcoming events."""
    future_events = engine.fetch_gov_events()
    recommendations = engine.predict_recommendation('default', future_events)
    if recommendations is None:
        raise HTTPException(status_code=404, detail="Model not trained")
    return {"client_id": 'default', "recommendations": recommendations}
