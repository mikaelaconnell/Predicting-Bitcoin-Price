## MacroBTC Forecasting App

## Project Overview
**MacroBTC** is a Streamlit web application that predicts Bitcoin prices using macroeconomic indicators such as inflation, interest rates, and global money supply. It blends modern machine learning with economic fundamentals to offer an intuitive, interactive crypto forecasting tool.

The motivation behind this project is my interest in understanding how monetary policy, inflation, and liquidity affect Bitcoin’s price movements, and using machine learning to make predictions. The domain interest this project focuses on is financial markets and cryptocurrency analysis. There are many other crypto forecasting models, but most rely on technical analysis rather than macroeconomic fundamentals, making this project a unique approach. 

---

## Problem Statement
Bitcoin price movements are notoriously volatile and often driven by speculation. Many existing models focus only on crypto-specific indicators, ignoring the potential influence of real-world economic conditions. This project explores the question:

> *Can macroeconomic indicators improve the accuracy of Bitcoin price predictions?*

---

## 💡 Solution
This app combines data from both crypto markets and traditional economic sources to build a machine learning model (XGBoost Regressor) capable of forecasting Bitcoin prices. Users can explore data relationships, generate predictions, and view model results in a clean, user-friendly dashboard.

---

## Features
- Predict future Bitcoin prices based on user-adjustable macro inputs  
- Explore historical data & macroeconomic trends  
- View correlation matrices and feature importance  
- Built with XGBoost for time series prediction  
- Interactive Streamlit dashboard

---

## Data Sources & Preprocessing

**Crypto Market Data**
- Bitcoin Price
- Trading Volume
- Market Dominance  
(Sourced from Federal Reserve Economic Data, Trading View, CoinMarketCap, & Yahoo Finance)

**Macroeconomic Indicators**
- U.S. M2 & Global M2 Money Supply
- CPI & PPI (Inflation)
- Federal Funds Rate
- Stock Indices: S&P 500, Nasdaq

**Data Cleaning & Scaling**
- Aligned datasets to daily frequency  
- Interpolated missing values  
- Removed outliers in crypto metrics  
- Scaled features using MinMaxScaler  

---

## Machine Learning Model

- **Model**: XGBoost Regressor  
- **Target Variable**: Monthly average BTC closing price  
- **Evaluation Metrics**: RMSE, R² Score  
- **Feature Importance** visualized using model outputs  
- Captures both macro trends and market-specific volatility  

---
