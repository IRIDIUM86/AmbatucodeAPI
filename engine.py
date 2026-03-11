import requests
import os
import pandas as pd
import xgboost as xgb
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class GovernmentDataEngine:
    def __init__(self):
        # Real government API (Malaysian public holidays via Google Calendar)
        self.GOV_API_URL = "https://www.googleapis.com/calendar/v3/calendars/en.malaysia%23holiday%40group.v.calendar.google.com/events?key=YOUR_API_KEY&timeMin={}&timeMax={}&singleEvents=true"
        self.MODEL_DIR = "models"
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        self.base_model_path = f"{self.MODEL_DIR}/base_food_bev_model.json"
        self.product_mappings = {}
        self.event_mappings = {}
        # Predefined Food and Beverages products
        self.default_products = ['beverages', 'snacks', 'meals', 'desserts', 'bakery']
        # Load or create base model
        self.load_base_model()

    def load_base_model(self):
        if os.path.exists(self.base_model_path):
            self.base_model = xgb.XGBRegressor()
            self.base_model.load_model(self.base_model_path)
        else:
            # Create a simple base model (train on synthetic data if needed)
            # For now, placeholder; in production, pre-train on historical data
            self.base_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)

    def fetch_gov_events(self, start_date=None, end_date=None):
        if not start_date:
            start_date = datetime.now().isoformat() + 'Z'
        if not end_date:
            end_date = (datetime.now() + pd.Timedelta(days=365)).isoformat() + 'Z'
        url = self.GOV_API_URL.format(start_date, end_date)
        response = requests.get(url)
        data = response.json().get('items', [])
        events = [{'date': item['start']['date'], 'event_type': item['summary']} for item in data]
        df = pd.DataFrame(events)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def train_for_client(self, client_id: str, sales_data: pd.DataFrame):
        # sales_data should already be a DataFrame (e.g. read from CSV)
        sales_data['date'] = pd.to_datetime(sales_data['date'])
        sales_data['day_of_week'] = sales_data['date'].dt.dayofweek
        
        events = self.fetch_gov_events()
        df = pd.merge(sales_data, events, on='date', how='left')
        df['is_event'] = df['event_type'].notna().astype(int)
        
        # Domain-specific: Map to Food and Beverages categories
        df['product'] = df['product'].apply(lambda x: x if x in self.default_products else 'other')
        product_cat = pd.Categorical(df['product'], categories=self.default_products + ['other'])
        df['product'] = product_cat.codes
        self.product_mappings[client_id] = product_cat.categories
        
        event_cat = pd.Categorical(df['event_type'].fillna('none'), categories=['none', 'holiday', 'festival'])  # Example events
        df['event_type'] = event_cat.codes
        self.event_mappings[client_id] = event_cat.categories
        
        X = df[['event_type', 'is_event', 'day_of_week', 'product']]
        y = df['sales_volume']
        
        # Fine-tune base model
        model = self.base_model
        model.fit(X, y)  # Fine-tune on client data
        model.save_model(f"{self.MODEL_DIR}/{client_id}.json")
        return True

    def test_model(self, client_id: str, test_data: pd.DataFrame):
        model_path = f"{self.MODEL_DIR}/{client_id}.json"
        if not os.path.exists(model_path):
            return None
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        # Simple test: Predict and compute RMSE
        predictions = model.predict(test_data[['event_type', 'is_event', 'day_of_week', 'product']])
        rmse = ((test_data['sales_volume'] - predictions) ** 2).mean() ** 0.5
        return {'rmse': rmse, 'predictions': predictions.tolist()}

    def predict_recommendation(self, client_id: str, future_events: pd.DataFrame):
        model_path = f"{self.MODEL_DIR}/{client_id}.json"
        if not os.path.exists(model_path) or client_id not in self.product_mappings:
            return None
        
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        
        future_events = future_events.copy()
        future_events['date'] = pd.to_datetime(future_events['date'])
        future_events['day_of_week'] = future_events['date'].dt.dayofweek
        future_events['is_event'] = 1
        
        event_cat = pd.Categorical(future_events['event_type'], categories=self.event_mappings[client_id])
        future_events['event_type'] = event_cat.codes
        
        products = self.product_mappings[client_id]
        recommendations = []
        for i, product in enumerate(products):
            X_future = future_events.copy()
            X_future['product'] = i
            predictions = model.predict(X_future[['event_type', 'is_event', 'day_of_week', 'product']])
            avg_pred = predictions.mean()
            recommendations.append({'product': product, 'predicted_sales': avg_pred})
        
        recommendations.sort(key=lambda x: x['predicted_sales'], reverse=True)
        return recommendations[:5]  # Top 5 with scores