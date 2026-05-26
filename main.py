import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import yfinance as yf

# Initialize the App
app = FastAPI(title="Nike Stock Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the AI and Scaler
model = load_model('nike_lstm_model.keras')
scaler = joblib.load('nike_scaler.pkl')

@app.get("/")
def home():
    return {"message": "Welcome to the Nike Stock Predictor API (Live Data Version)!"}


# ==========================================
# FEATURE 1: The Historical Backtester (LIVE YAHOO DATA)
# ==========================================
@app.get("/predict_historical")
def predict_historical(target_date: str):
    try:
        # 1. Connect to Yahoo Finance instead of the CSV!
        nke = yf.Ticker("NKE")
        
        # 2. Download all history up to the target date
        df = nke.history(start="2000-01-01", end=target_date)
        
        if df.empty or len(df) < 60:
            return {"status": "error", "message": "Market was closed on this date, or not enough history."}
            
        # 3. Grab the 60 days of closing prices IMMEDIATELY BEFORE the target date
        past_60_days = df.iloc[-60:]['Close'].values
        
        # 4. Fetch the actual price on the target date separately to grade the AI
        target_day_data = nke.history(start=target_date, end=pd.to_datetime(target_date) + pd.Timedelta(days=1))
        
        if target_day_data.empty:
            return {"status": "error", "message": "Market was closed on your selected date. Pick a weekday!"}
            
        actual_price = float(target_day_data.iloc[0]['Close'])
        
        # 5. Prepare data for the AI
        past_60_days_scaled = scaler.transform(past_60_days.reshape(-1, 1))
        X_input = np.reshape(past_60_days_scaled, (1, 60, 1))
        
        # 6. Ask the AI for its prediction
        predicted_scaled = model.predict(X_input)
        predicted_price = float(scaler.inverse_transform(predicted_scaled)[0][0])
        
        return {
            "status": "success",
            "date": target_date,
            "predicted_price": round(predicted_price, 2),
            "actual_price": round(actual_price, 2),
            "error_margin": round(abs(actual_price - predicted_price), 2)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ==========================================
# FEATURE 2: The Live Future Forecaster
# ==========================================
@app.get("/predict_live")
def predict_live():
    try:
        # 1. Connect to Yahoo Finance and grab Nike (NKE)
        nke = yf.Ticker("NKE")
        
        # 2. Download the last 3 months of real-world data
        hist = nke.history(period="3mo")
        
        if len(hist) < 60:
            return {"status": "error", "message": "Could not fetch enough live data."}
            
        # 3. Isolate the absolute most recent 60 days of Close prices!
        last_60_days = hist['Close'].values[-60:]
        
        # Get today's actual date and closing price for the UI
        latest_date = hist.index[-1].strftime('%Y-%m-%d')
        latest_price = float(hist['Close'].iloc[-1])
        
        # 4. Prepare data for our AI
        last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
        X_input = np.reshape(last_60_days_scaled, (1, 60, 1))
        
        # 5. Make the prediction
        predicted_scaled = model.predict(X_input)
        predicted_price = float(scaler.inverse_transform(predicted_scaled)[0][0])
        
        # --- NEW DATE LOGIC ---
        # Calculate the next valid trading day's calendar date
        today = datetime.date.today()
        if today.weekday() == 4: # If today is Friday, next market day is Monday (+3 days)
            next_day = today + datetime.timedelta(days=3)
        elif today.weekday() == 5: # If today is Saturday, next market day is Monday (+2 days)
            next_day = today + datetime.timedelta(days=2)
        else: # Sunday through Thursday, next market day is tomorrow (+1 day)
            next_day = today + datetime.timedelta(days=1)
            
        predicted_date = next_day.strftime('%Y-%m-%d')
        
        return {
            "status": "success",
            "latest_trading_day": latest_date,
            "latest_actual_price": round(latest_price, 2),
            "predicted_tomorrow_price": round(predicted_price, 2),
            "predicted_date": predicted_date  # <-- Sending the exact date to the UI!
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}