import uvicorn
import csv
import io
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Annotated, Optional, List
from engine import DataEngine

app = FastAPI(
    title="CSV Processing API",
    description="An API that accepts JSON and CSV files.",
    version="1.1.0"
)

class Item(BaseModel):
    name: str
    value: float
    description: Optional[str] = None

@app.get("/")
def read_root():
    return {"status": "Online", "message": "Welcome. Use /docs to test CSV uploads."}

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    # Validate file extension
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .csv file.")

    try:
        # Read the file content
        content = await file.read()
        
        # Decode the bytes to string and use io.StringIO to make it readable by the csv module
        csv_data = content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(csv_data))
        
        # Convert CSV rows into a list of dictionaries
        data = [row for row in reader]
        
        return {
            "filename": file.filename,
            "row_count": len(data),
            "data_preview": data[:5],  # Return first 5 rows as a preview
            "status": "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@app.post("/train-model")
async def train_model(product_var_name: Annotated[str, Form()], date_var_name: Annotated[str, Form()], file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .csv file.")

    temp_path = None
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(content.decode('utf-8'))
            temp_path = temp_file.name

        engine = DataEngine(temp_path, product_var_name, date_var_name)
        success = engine.train_model(engine.df)

        if success:
            return {"status": "Training completed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Training failed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/predict-recommendation")
async def predict_recommendation(product_var_name: Annotated[str, Form()], date_var_name: Annotated[str, Form()], file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .csv file.")

    temp_path = None
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(content.decode('utf-8'))
            temp_path = temp_file.name

        engine = DataEngine(temp_path, product_var_name, date_var_name)
        recommendations = engine.predict_recommendation(engine.df)

        if recommendations:
            return {"recommendations": recommendations}
        else:
            return {"message": "No recommendations available"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)