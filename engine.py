import requests
import os
import pandas as pd
import xgboost as xgb
import joblib
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
import time
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

app = FastAPI()
load_dotenv()

class DataEngine:
    def __init__(self, filename: str, product_var_name: str, date_var_name: str):
        # Initialize API parameters and load product mappings
        self.base_url = "https://app.ticketmaster.com/discovery/v2/"
        self.api_key = os.getenv("EVENT_API_KEY")

        self.product_var_name = product_var_name 
        self.date_var_name = date_var_name

        self.df = pd.read_csv(filename)
        unique_values = self.df[product_var_name].unique()
        self.product_mappings = {value: i for i, value in enumerate(unique_values)}
        self.num_products = len(self.product_mappings)

        # Load base model
        self.MODEL_DIR = "/tmp/models"
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        self.base_model_path = f"{self.MODEL_DIR}/base_food_bev_model.json"
        self.load_base_model()

    def load_base_model(self):
        if os.path.exists(self.base_model_path):
            print(f"Loading existing model from {self.base_model_path}")
            self.base_model = xgb.XGBClassifier()
            self.base_model.load_model(self.base_model_path)

        else:
            self.base_model = xgb.XGBClassifier(n_estimators=1000,
                                                learning_rate=0.05,
                                                max_depth=6,
                                                objective='multi:softprob',
                                                num_class=self.num_products
                                                )

    def fit_date(self, df: pd.DataFrame):
        # 1. Ensure the client's date column is in datetime format
        df[self.date_var_name] = pd.to_datetime(
            df[self.date_var_name].astype(str).str.strip(),
            errors="coerce",
            dayfirst=True,
        ).dt.normalize()

        # If there are no valid dates, skip enrichment and return the original DataFrame.
        if df[self.date_var_name].dropna().empty:
            df["event_name"] = "None"
            df["event_type"] = "None"
            return df

        # 2. Identify the range of dates in the client's data to limit API calls
        min_date = df[self.date_var_name].min().strftime('%Y-%m-%dT00:00:00Z')
        max_date = df[self.date_var_name].max().strftime('%Y-%m-%dT23:59:59Z')

        event_list = []

        # 3. Fetch events from API within this specific timeframe
        for page in range(5):
            params = {
                "apikey": self.api_key,
                "startDateTime": min_date,
                "endDateTime": max_date,
                "size": 200,
                "page": page,
                "sort": "date,asc"
            }

            try:
                response = requests.get(f"{self.base_url}events.json", params=params)

                if response.status_code == 429:
                    print(f"Rate limit hit. Sleeping for 0.5 seconds...")
                    time.sleep(0.5)
                    response = requests.get(f"{self.base_url}events.json", params=params)

                response.raise_for_status()
                api_data = response.json()
                
                if "_embedded" in api_data:
                    for event in api_data["_embedded"]["events"]:
                        # Extract the date only (YYYY-MM-DD) for merging
                        raw_date = event.get("dates", {}).get("start", {}).get("localDate")
                        if raw_date:
                            # Collect name and the segment/type ID
                            classifications = event.get("classifications", [{}])
                            segment_name = classifications[0].get("segment", {}).get("name", "Other")
                        
                            event_list.append({
                                "api_event_date": pd.to_datetime(raw_date).normalize(),
                                "event_name": event.get("name"),
                                "event_type": segment_name
                            })

                if not event_list:
                    df["event_name"] = "None"
                    df["event_type"] = "None"
                    return df
                
                # 4. Create a reference DataFrame from the API results
                api_df = pd.DataFrame(event_list)

                # 5. AGGREGATE: Join multiple events/types into single strings
                # This ensures all event types for that day are included.
                api_df_cleaned = api_df.groupby("api_event_date").agg({
                    "event_name": lambda x: "|".join(x.unique()),
                    "event_type": lambda x: "|".join(x.unique()) 
                }).reset_index()

                # 6. Merge: Left join ensures we keep all original client data
                enriched_df = pd.merge(
                    df, 
                    api_df_cleaned, 
                    left_on=self.date_var_name, 
                    right_on="api_event_date", 
                    how="left"
                )

                # 7. Final Cleanup
                enriched_df["event_type"] = enriched_df["event_type"].fillna("None")
                enriched_df["event_name"] = enriched_df["event_name"].fillna("None")
                enriched_df.drop(columns=["api_event_date"], inplace=True)

                return enriched_df

            except Exception as e:
                print(f"Error during date fitting: {e}")
                df["event_name"] = "None"
                df["event_type"] = "None"
                return df

    def train_model(self, df: pd.DataFrame):
        df = self.fit_date(df) # Enrich the data with event information based on the date

        # 1. Feature Engineering: Handle the 'event_type' column
        # Split the pipe-separated string back into a list for binarization
        df['event_list'] = df['event_type'].apply(lambda x: x.split('|') if x != "None" else [])
        
        mlb = MultiLabelBinarizer()
        event_type_encoded = mlb.fit_transform(df['event_list'])
        event_type_df = pd.DataFrame(event_type_encoded, columns=mlb.classes_, index=df.index)

        # 2. Feature Engineering: Handle 'event_name' using TF-IDF
        tfidf = TfidfVectorizer(max_features=100, stop_words=None)
        event_names_tfidf = tfidf.fit_transform(df['event_name'].fillna('None'))
        feature_names = tfidf.get_feature_names_out()
        event_names_df = pd.DataFrame(
            event_names_tfidf.toarray(), 
            columns=[f"tfidf_{name}" for name in feature_names], 
            index=df.index
        )

        # 3. Combine original features with new event features
        X = df.drop(columns=[self.product_var_name, self.date_var_name, 'event_name', 'event_type', 'event_list'])
        X = pd.concat([X, event_type_df, event_names_df], axis=1)
        
        # 4. Target Encoding
        y = df[self.product_var_name].map(self.product_mappings)

        # 5. Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 6. Train the Model
        print(f"Training on {len(X_train)} rows with {X_train.shape[1]} features...")
        self.base_model.fit(X_train, 
                            y_train, 
                            eval_set=[(X_test, y_test)],
                            verbose=False
                            )
        
        self.base_model.save_model(f"{self.base_model_path}")
        joblib.dump(mlb, f"{self.MODEL_DIR}/mlb.joblib")
        joblib.dump(tfidf, f"{self.MODEL_DIR}/tfidf.joblib")

        # 7. Evaluate the Model
        y_pred = self.base_model.predict(X_test)
        print(classification_report(y_test, y_pred))
        return True
    
    def get_future_events(self, df: pd.DataFrame):
        # 1. Get the max date and format it correctly for the API
        max_date_dt = pd.to_datetime(
            df[self.date_var_name].astype(str).str.strip(),
            errors="coerce",
            dayfirst=True,
        ).max()

        if pd.isna(max_date_dt):
            return pd.DataFrame()

        start_date_str = max_date_dt.strftime('%Y-%m-%dT23:59:59Z')

        future_event_list = []

        # 2. Use 'startDateTime' (the correct API key)
        params = {
            "apikey": self.api_key,
            "startDateTime": start_date_str, 
            "size": 100,
            "sort": "date,asc"
        }

        try:
            response = requests.get(f"{self.base_url}events.json", params=params)
            response.raise_for_status()
            api_data = response.json()
            
            if "_embedded" in api_data:
                for event in api_data["_embedded"]["events"]:
                    raw_date = event.get("dates", {}).get("start", {}).get("localDate")
                    if raw_date:
                        classifications = event.get("classifications", [{}])
                        segment_name = classifications[0].get("segment", {}).get("name", "Other")
                    
                        future_event_list.append({
                            # Keep the column name consistent with your date_var_name for easier merging later
                            self.date_var_name: pd.to_datetime(raw_date).normalize(),
                            "event_name": event.get("name"),
                            "event_type": segment_name
                        })

            # 3. Aggregate results (In case multiple events happen on the same future day)
            if not future_event_list:
                return pd.DataFrame()

            future_events_df = pd.DataFrame(future_event_list)
            
            future_events_df = future_events_df.groupby(self.date_var_name).agg({
                "event_name": lambda x: "|".join(x.unique()),
                "event_type": lambda x: "|".join(x.unique())
            }).reset_index()

            return future_events_df

        except Exception as e:
            print(f"Error fetching future events: {e}")
            return pd.DataFrame() # Return empty DataFrame on error to avoid breaking the prediction flow
    
    def predict_recommendation(self, df: pd.DataFrame):
        if not os.path.exists(self.base_model_path):
            return None # No model available
        
        # 1. Load the model and translators (MLB/TFIDF)
        model = xgb.XGBClassifier()
        model.load_model(self.base_model_path)

        try:
            mlb = joblib.load(f"{self.MODEL_DIR}/mlb.joblib")
            tfidf = joblib.load(f"{self.MODEL_DIR}/tfidf.joblib")
        except:
            print("Translators (MLB/TFIDF) not found. Train the model first.")
            return None
        
        future_events = self.get_future_events(df)
        if future_events.empty:
            return None # No future events to predict on
        
        # 2. Feature Engineering (The "Translation" Phase)
        event_list = future_events['event_type'].apply(lambda x: x.split('|') if x != "None" else [])
        type_encoded = mlb.transform(event_list) # Use transform only
        type_df = pd.DataFrame(type_encoded, columns=mlb.classes_, index=future_events.index)

        # 3. Process Event Names
        name_tfidf = tfidf.transform(future_events['event_name'].fillna('None'))
        feature_names = tfidf.get_feature_names_out()
        name_df = pd.DataFrame(
            name_tfidf.toarray(), 
            columns=[f"tfidf_{name}" for name in feature_names], 
            index=future_events.index
        )

        # 4. Prepare Final Feature Matrix (X)
        # Add dummy original features (since we don't have them for future)
        dummy_features = pd.DataFrame({
            'Quantity': [0] * len(future_events),
            'Price Per Unit': [0] * len(future_events),
            'Total Spent': [0] * len(future_events),
            'Payment Method': [0] * len(future_events),
            'Location': [0] * len(future_events)
        }, index=future_events.index)
        
        X_final = pd.concat([dummy_features, type_df, name_df], axis=1)

        # 5. Generate Predictions
        predictions = model.predict(X_final)
        probs = model.predict_proba(X_final) # Get probabilities for all classes (optional, if you want "top 5")

        # 6. Map IDs back to Product Names
        inv_map = {v: k for k, v in self.product_mappings.items()}
        
        # 7. Add predictions to the future_events dataframe for context
        future_events['recommended_product'] = [inv_map.get(p, "Unknown") for p in predictions]
        
        # 8. Return the top results
        return future_events[[self.date_var_name, 'event_name', 'recommended_product']].to_dict('records')