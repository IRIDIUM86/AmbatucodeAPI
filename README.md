# Ambatucode API

A **free** API providing machine-learning-driven product recommendations for Food and Beverages SMEs. Clients upload their sales CSV and the service combines it with public event data from Ticketmaster to suggest what to sell during upcoming events.

## Features

- Free to use with no authentication or billing
- Upload client-specific sales data via CSV
- Automatically train an XGBoost model using event data from Ticketmaster API
- Receive product recommendations for future events

## Setup

1. Install dependencies: `pip install -r requirements.txt` or `python -m pip install -r requirements.txt`
2. Set up environment: Create a `.env` file with `EVENT_API_KEY=your_ticketmaster_api_key`
3. Run the server: `uvicorn main:app --reload`

## API Endpoints

- `POST /upload-csv` – Upload a CSV file to preview its data (returns row count and first 5 rows)
- `POST /train-model` – Upload a CSV file, provide `product_var_name` and `date_var_name` as form data, to train the XGBoost model
- `POST /predict-recommendation` – Upload a CSV file, provide `product_var_name` and `date_var_name` as form data, to get product recommendations for upcoming events
- `POST /delete-models-dir` – Delete the local `models/` folder (removes trained model and translators). Use this if you want to clear the cached model and force retraining.

No API key or authentication is required for the endpoints.

## Data Format

Sales data CSV should have columns including:
- `Item` (or specified product column): The product identifier
- `Transaction Date` (or specified date column): The date of the transaction
- Other features like `Quantity`, `Price Per Unit`, `Total Spent`, `Payment Method`, `Location`

The API fetches event data from Ticketmaster and enriches the data for training and prediction.

## Notes

- The model is trained per request and saved in the `models/` directory.
- Uses Ticketmaster Discovery API for event data; ensure your API key is valid.
- For production, consider persistent storage and error handling improvements.