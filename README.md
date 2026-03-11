# Ambatucode API

A paid API for machine learning-based product recommendations for Food and Beverages SMEs based on government events.

## Features

- Upload sales data (CSV)
- Train XGBoost model on sales data merged with government event data
- Get product recommendations for upcoming events

## Setup

1. Install dependencies: `pip install -r requirements.txt` or `python -m pip install -r requirements.txt`
2. Run the server: `uvicorn main:app --reload`

## API Endpoints

- `POST /upload-data/{client_id}`: Upload sales CSV data
- `POST /train/{client_id}`: Start training the model
- `GET /recommend/{client_id}`: Get product recommendations

All endpoints require `X-API-Key` header with a valid API key.

## Data Format

Sales data CSV should have columns: `date`, `product`, `sales_volume`

Government API should return JSON with `date` and `event_type` fields.

## Notes

- This is a foundational codebase. In production, add proper database, authentication, billing, etc.
- Government API URL is placeholder; replace with actual API.
- For direct DB connection, add endpoints to accept DB credentials and query data.