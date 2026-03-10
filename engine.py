import requests
import os
import pandas as pd
import xgboost as xgb
from datetime import datetime

class GovernmentDataEngine:
    def __init__(self):
        self.GOV_API_URL = "https://api.gov.my/events"  # Placeholder URL
        self.MODEL_DIR = "models"
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        self.product_mappings = {}
        self.event_mappings = {}

    def fetch_gov_events(self):
        response = requests.get(self.GOV_API_URL)
        data = response.json()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def train_for_client(self, client_id: str, sales_data: pd.DataFrame):
        # Assume sales_data has columns: date, product, sales_volume
        sales_data['date'] = pd.to_datetime(sales_data['date'])
        sales_data['day_of_week'] = sales_data['date'].dt.dayofweek
        
        # 1. Get Event Data
        events = self.fetch_gov_events()
        
        # 2. Merge Sales with Events on Date
        df = pd.merge(sales_data, events, on='date', how='left')
        df['is_event'] = df['event_type'].notna().astype(int)
        
        # 3. Preprocessing
        event_cat = df['event_type'].fillna('none').astype('category')
        df['event_type'] = event_cat.cat.codes
        self.event_mappings[client_id] = event_cat.cat.categories
        
        product_cat = df['product'].astype('category')
        df['product'] = product_cat.cat.codes
        self.product_mappings[client_id] = product_cat.cat.categories
        
        X = df[['event_type', 'is_event', 'day_of_week', 'product']]
        y = df['sales_volume']
        
        # 4. XGBoost Training
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X, y)
        
        # 5. Save model
        model.save_model(f"{self.MODEL_DIR}/{client_id}.json")
        return True

    def predict_recommendation(self, client_id: str, future_events: pd.DataFrame):
        model_path = f"{self.MODEL_DIR}/{client_id}.json"
        if not os.path.exists(model_path) or client_id not in self.product_mappings:
            return None
        
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        
        # Assume future_events has date, event_type
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
            recommendations.append((product, avg_pred))
        
        # Sort by predicted sales descending
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [prod for prod, _ in recommendations[:5]]  # Top 5 products