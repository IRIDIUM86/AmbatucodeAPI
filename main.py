import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# 1. Initialize the App
app = FastAPI(
    title="My Python API",
    description="A server to allow other folders to communicate with my logic.",
    version="1.0.0"
)

# 2. Define Data Models (Ensures the "User" sends the right stuff)
class Item(BaseModel):
    name: str
    value: float
    description: Optional[str] = None

# 3. The "Home" Route (Prevents the 404 error you saw)
@app.get("/")
def read_root():
    return {
        "status": "Online",
        "message": "Welcome to the API. Go to /docs for interactive testing."
    }

# 4. A Sample "Processing" Route
@app.post("/process")
def process_data(item: Item):
    # This is where your actual Python logic lives
    # For example: calculation or file manipulation
    doubled_value = item.value * 2
    
    return {
        "received_name": item.name,
        "result": doubled_value,
        "status": "Success"
    }

# 5. The "No-Terminal-Error" Launcher
if __name__ == "__main__":
    # This bypasses the 'uvicorn not recognized' issue
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)