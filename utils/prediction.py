import pandas as pd
import numpy as np
import joblib

# Load model once (optional: make this global for efficiency)
model_path = 'model/bitcoin_model.pkl'

def make_prediction(df, days):
    # Just a dummy model prediction for now
    # Replace this with actual preprocessing and prediction logic
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        raise Exception("Model file not found. Make sure 'bitcoin_model.pkl' is in the model/ directory.")

    last_price = df['Close'].iloc[-1]
    
    # Generate dummy predictions
    predicted_prices = [last_price + np.random.randn() * 500 for _ in range(days)]
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days)

    prediction_df = pd.DataFrame({
        'predicted_price': predicted_prices
    }, index=future_dates)

    return prediction_df