import yfinance as yf
import pandas as pd

def load_data(interval='Daily', ticker='BTC-USD'):
    interval_map = {
        'Daily': '1d',
        'Weekly': '1wk',
        'Monthly': '1mo'
    }
    period = '1y'  # default: load last 1 year of data
    yf_interval = interval_map.get(interval, '1d')

    df = yf.download(ticker, period=period, interval=yf_interval)
    df.dropna(inplace=True)
    return df