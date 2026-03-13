# Ambatucode API

A **free** API providing machine‑learning‑driven product recommendations for Food and Beverages SMEs. Clients upload their sales CSV and the service combines it with public government event data to suggest what to sell during upcoming events.

## Features

- Free to use with no authentication or billing
- Upload client-specific sales data via CSV
- Automatically train an XGBoost model per client using government event data
- Receive top‑product recommendations for future events

## Setup

1. Install dependencies: `pip install -r requirements.txt` or `python -m pip install -r requirements.txt`
2. Run the server: `uvicorn main:app --reload`

## API Endpoints

- `POST /upload-data/{client_id}` – upload a CSV containing `date`, `product`, and `sales_volume` columns
- `POST /train/{client_id}` – trigger background training for that client’s data
- `POST /test/{client_id}` – optionally upload a test CSV to evaluate model accuracy
- `GET /recommend/{client_id}` – return top product recommendations for upcoming events

No API key or authentication is required; the `client_id` is simply a namespace to separate different users or experiments.

## Data Format

Sales data CSV should have columns: `date`, `product`, `sales_volume`

Government API should return JSON with `date` and `event_type` fields.

## Notes

- The current implementation uses in‑memory storage and expects CSVs; consider persistent storage for production.
- Government API URL is a placeholder; configure your own API key in `.env` (e.g. Google Calendar public holidays).
- `client_id` can be any identifier and does not imply authentication or payment.