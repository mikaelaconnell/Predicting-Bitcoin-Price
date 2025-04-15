import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import joblib
from PIL import Image
from utils.data_loader import load_data
from utils.prediction import make_prediction
from pycoingecko import CoinGeckoAPI
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import yfinance as yf
import os

# Set the streamlit app layout to wide
st.set_page_config(layout="wide")

# Import custom Google Font
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# --- CSS Styling ---
st.markdown("""
    <style>
    /* Make background cover the full app */
    html, body, .block-container {
        background-color: #f5f3ff !important;  /* Light purple */
        color: #2e1065 !important;            /* Dark purple text */
        font-family: 'Space Grotesk', sans-serif !important;
    }

    /* Navbar styling */
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 2rem;
        background-color: #ede9fe;
        border-bottom: 2px solid #d8b4fe;
        margin-bottom: 1rem;
    }

    .navbar img {
        height: 40px;
        }

    .navbar a {
        text-decoration: none;
        color: #5b21b6;
        font-weight: 700;
        font-size: 1.5rem;
    }

    /* Tabs (Top Nav Tabs) */
    .css-1hynsf2, .stTabs {
        background-color: #ede9fe;
        padding: 0.5rem 1rem;
        border-bottom: 1px solid #d8b4fe;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ddd6fe;
        border-radius: 12px;
        padding: 1em;
        margin: 1em 0;
        color: #4c1d95;
        box-shadow: 0 0 8px rgba(90, 30, 180, 0.1);
    }

    /* Buttons */
    .stButton > button {
        background-color: #8b5cf6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75em 1.5em;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.4);
    }

    .stButton > button:hover {
        background-color: #7c3aed;
        transform: scale(1.05);
        }

    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("MacroBTC Insights")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Prediction Results", "Data Exploration", "Data Sources"])   
tab_dict = {
    "Overview": tab1,
    "Prediction Results": tab2,
    "Data Exploration": tab3,
    "Data Sources": tab4
}

with tab_dict["Overview"]:
        # --- Page Layout ---
        left_col, right_col = st.columns(2)
        with left_col:
            st.subheader("What is MacroBTC?")
            st.write("MacroBTC blends Bitcoin analytics with global economic indicators to predict market behavior. We analyze trends in inflation, interest rates, M2 money supply, and more.")
            st.markdown("""
                ### About This App
                This Bitcoin Price Prediction App is a dynamic forecasting tool powered by machine learning and macroeconomic analysis.
                Designed for analysts, crytpto enthusiasts, and curious investors, the app blends real-time market data with historical 
                economic indicators to deliver meaningful insights into Bitcoin's price movements.
                    
                - **Real-time Bitcoin Price Tracking:** Visualize live BTC price movements alongside economic factors
                - **Predictive Analytics:** Leverage machine learning to forecast Bitcoin prices
                - **Data Exploration:** Explore the relationship between Bitcoin prices and key economic indicators through interactive charts
                - **Customizable Forecast Assumptions:** Tailor the model's assumptions to fit your investment strategy
                - **Data Sources:** Comprehensive data from reliable sources like CoinGecko, Yahoo Finance, and the Federal Reserve 
            """)

        with right_col:
            cg = CoinGeckoAPI()
            btc_data = cg.get_price(ids='bitcoin', vs_currencies='usd')
            latest_price = btc_data['bitcoin']['usd']
            st.metric(label="Latest BTC Price", value=f"${latest_price:,.2f}")
            # st.image("image/bitcoin4.jpg", width=650)
            # Title
            st.subheader("Live Bitcoin Price (Last 30 Days)")
            # Initialize CoinGecko API client
            cg = CoinGeckoAPI()

            # Fetch Bitcoin data
            bitcoin_data = cg.get_coin_market_chart_range_by_id(id='bitcoin', vs_currency='usd', 
                                                        from_timestamp=int((datetime.datetime.now() - datetime.timedelta(days=30)).timestamp()),
                                                        to_timestamp=int(datetime.datetime.now().timestamp()))

            # Convert to DataFrame
            df = pd.DataFrame(bitcoin_data['prices'], columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Plot the Bitcoin price chart
            fig = px.line(df, x='timestamp', y='price')

            # Display the plot as an interactive graph
            st.plotly_chart(fig)
    

with tab_dict["Prediction Results"]:
      # Construct a relative path to the model
      model_path = os.path.join('model', 'bitcoin_model.pkl')
      model = joblib.load(model_path)

      # --- Sidebar ---
      st.sidebar.image("image/Macro_BTC.png", width=270)
      st.sidebar.write("Choose the prediction range and time interval below:")
      st.sidebar.write("After running the prediction, check the Predictive Analytics tab for the results!")
      days = st.sidebar.number_input("Number of Days to Forecast", min_value=1, max_value=90, value=7)
      interval = st.sidebar.selectbox("Select Interval", ["Daily", "Weekly", "Monthly"])

      if st.sidebar.button("Run Prediction"):
            run_prediction = True
      else:
            run_prediction = False

      # --- Load Data ---
      data = load_data(interval=interval)
      latest_price = float(data['Close'].iloc[-1])

      # --- Run Prediction ---
      if run_prediction:
        with st.spinner("Running prediction..."):
            col1, col2 = st.columns(2)
                
            with col1:
                cg = CoinGeckoAPI()
                btc_data = cg.get_price(ids='bitcoin', vs_currencies='usd')
                latest_price = btc_data['bitcoin']['usd']
                st.metric(label="Current BTC Price", value=f"${latest_price:,.2f}")
                
                prediction_df = make_prediction(data, days)
                predicted_price = float(prediction_df['predicted_price'].values[-1])

                # --- Highlight Metrics ---
                st.subheader("Market Insights")

                price_change = predicted_price - latest_price
                trend = "Uptrend 📈 " if price_change > 0 else "Downtrend 📉"
                st.metric(label="Price Change", value=f"${price_change:,.2f}", delta=f"{price_change / latest_price * 100:.2f}%")
                st.metric(label="Take Profit", value=f"${predicted_price * 1.05:,.2f}")
                    
                    
            with col2:
                st.metric(label="Predicted Price", value=f"${predicted_price:,.2f}")
                st.subheader(" ")
                st.metric(label="BTC Trend", value=trend, delta=f"{price_change / latest_price * 100:.2f}%")
                st.metric(label="Stop Loss", value=f"${predicted_price * 0.95:,.2f}")

            # Plot predictions
            st.subheader("Predicted Bitcoin Prices")
            st.line_chart(prediction_df['predicted_price'])
            st.metric(label="Buy/Sell Suggestion", value="Buy" if price_change > 0 else "Sell")
            st.write("Note: The predictions are based on historical data and may not reflect actual future prices. The model uses XGBoost for predictions, and is trained on historical prices and macroeconomic indicators.")


with tab_dict["Data Exploration"]:
        st.header("Explore the Data")
        st.subheader("Key Economic Indicators:")
        st.write("""
                - **Global M2 Money Supply:** Total amount of money circulating in the global economy
                - **U.S. M2 Money Supply:** Indicator of domestic economic conditions and liquidity
                - **Inflation Rate (CPI, PPI):** Impact on purchasing power and overall economic stability
                - **Interest Rates (Federal Funds Rate):** Influences demand for investment and speculative assets like Bitcoin
                - **Stock Market Indices (S&P 500, Nasdaq):** Reflects overall market sentiment and investor behavior
                - **Bitcoin Trading Volume:** Measures the market activity and dominance of Bitcoin over other cryptocurrencies
            """)

        # Path to your CSV file
        csv_file_path = 'bitcoin_final_filled.csv'

        # Load the CSV file
        data = pd.read_csv(csv_file_path)
        
        # Ensure that the 'Date' column is in datetime format
        data['Date'] = pd.to_datetime(data['date'])

        # Extract Bitcoin price and Global M2 data
        bitcoin_price = data['btc_price']
        global_m2 = data['global_m2']

        # Column map: display name → real column name in DataFrame
        feature_map = {
            'Global M2 Money Supply': 'global_m2',
            'U.S. M2 Money Supply': 'us_m2_scaled',
            'CPI (Inflation Rate)': 'cpi',
            'PPI (Producer Price Index)': 'ppi',
            'Federal Funds Rate': 'fed_funds_rate',
            'S&P 500 Index': 'sp500',
            'Nasdaq Index': 'nasdaq',
            'Bitcoin Trading Volume': 'btc_volume'
        }

        # Selectbox with display names
        selected_display_name = st.selectbox(
            '### Select an economic indicator to compare to Bitcoin Price:',
            list(feature_map.keys())
        )

        # Use mapping to get actual column name
        selected_feature_column = feature_map[selected_display_name]

        # Create plot
        fig3 = go.Figure()

        fig3.add_trace(go.Scatter(
            x=data['date'], y=data['btc_price'], mode='lines', name='Bitcoin Price',
            line=dict(color='blue'))
    )

        fig3.add_trace(go.Scatter(
             x=data['date'], y=data[selected_feature_column], mode='lines', name=selected_display_name,
            line=dict(color='red'), yaxis="y2")
        )

        fig3.update_layout(
            title=f"{selected_display_name} vs. Bitcoin Price (2015–2025)",
            xaxis_title="Date",
            yaxis_title="Bitcoin Price (USD)",
            yaxis2=dict(
                title=selected_display_name,
                overlaying="y",
                side="right"
            ),
            template="plotly_dark",
            hovermode="x unified"
        )

        st.plotly_chart(fig3)

        # Plot of M2 vs. BTC price (108 days later)
        # Let user choose between Global or US M2
        m2_options = {
            'Global M2 Money Supply': 'global_m2',
            'U.S. M2 Money Supply': 'us_m2_scaled'
        }

        selected_m2_display = st.selectbox(
            "Select M2 Money Supply Type:", list(m2_options.keys())
        )
        selected_m2_column = m2_options[selected_m2_display]

        # Shift BTC price by -108 days to compare current M2 to future BTC price
        data_shifted = data.copy()
        data_shifted['btc_price_future'] = data_shifted['btc_price'].shift(-108)

        # Drop rows with NaN due to shifting
        data_shifted = data_shifted.dropna(subset=['btc_price_future'])

        # Create the figure
        fig = go.Figure()

        # Plot M2 on the left axis
        fig.add_trace(go.Scatter(
            x=data_shifted['date'], y=data_shifted[selected_m2_column],
            mode='lines', name=selected_m2_display, line=dict(color='purple')
        ))

        # Plot future BTC price on the right axis
        fig.add_trace(go.Scatter(
            x=data_shifted['date'], y=data_shifted['btc_price_future'],
            mode='lines', name='Bitcoin Price (108 Days Later)', line=dict(color='orange'),
            yaxis='y2'
        ))

        # Layout with dual y-axes
        fig.update_layout(
            title=f"{selected_m2_display} vs. Bitcoin Price (108 Days Later)",
            xaxis_title='Date',
            yaxis=dict(title=selected_m2_display),
            yaxis2=dict(
                title='Bitcoin Price (USD)',
                overlaying='y',
                side='right'
            ),
            template='plotly_dark',
            hovermode='x unified'
        )

        # Display the plot
        st.plotly_chart(fig)

        # --- Correlation Matrix ---
        st.subheader("Correlation Matrix")

        # Mapping from internal column names to display names
        feature_name_map = {
            'btc_price': 'Bitcoin Price',
            'global_m2': 'Global M2 Money Supply',
            'us_m2_scaled': 'U.S. M2 Money Supply',
            'sp500': 'S&P 500 Index',
            'nasdaq': 'Nasdaq Index',
            'fed_funds_rate': 'Federal Funds Rate',
            'cpi': 'CPI (Inflation Rate)',
            'ppi': 'PPI (Producer Price Index)',
            'btc_volume': 'Bitcoin Trading Volume'
        }

        # Select only relevant columns for correlation
        corr_df = data[list(feature_name_map.keys())]
        corr_matrix = corr_df.corr().round(2)

        # User-selected subset
        selected_features = st.multiselect(
            "Select Features to Display in the Correlation Matrix:",
            options=list(feature_name_map.keys()),
            default=list(feature_name_map.keys())
        )

        # Filter correlation matrix
        filtered_corr_matrix = corr_matrix[selected_features].loc[selected_features]

        # Map to display names for prettier axis labels
        display_labels = [feature_name_map[col] for col in selected_features]

        # Create Plotly annotated heatmap
        filtered_fig = ff.create_annotated_heatmap(
            z=filtered_corr_matrix.values,
            x=display_labels,
            y=display_labels,
            colorscale='Blues',
            font_colors=['white'] * len(display_labels),
            showscale=True,
            colorbar=dict(title='Correlation', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
        )

        filtered_fig.update_layout(
            xaxis_title="Features",
            yaxis_title="Features",
            template="plotly_dark",
            xaxis=dict(showgrid=True, showline=True),
            yaxis=dict(showgrid=True, showline=True),
            hovermode="closest",
            autosize=True
        )

        st.plotly_chart(filtered_fig)
    
        st.write("""
            ### How This Data Contributes to Bitcoin Price Prediction
            By combining multiple economic indicators like the M2 Money Supply, inflation rates, and stock market movements, this dataset allows us to capture the broader macroeconomic environment, which is crucial for predicting Bitcoin price fluctuations.
            Cryptocurrency markets are deeply influenced by broader economic trends. This app bridges the gap between crypto and macroeconomics
            by incorporating key financial indicators into the prediction model, providing more informed insights than purely technical approaches.
            """)


with tab_dict["Data Sources"]:
        st.header("Data Sources & Educational Resources")
        st.write("""
        - **Global M2 Money Supply:** [Trading View](https://www.tradingview.com/)
        - **U.S. M2 Money Supply:** [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/series/M2SL)
        - **Inflation Rate (CPI, PPI):** [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/series/CPIAUCSL)
        - **Interest Rates (Federal Funds Rate):** [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/series/FEDFUNDS)
        - **Stock Market Indices (S&P 500, Nasdaq):** [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/series/SP500)
        - **Bitcoin Trading Volume:** [Yahoo Finance](https://finance.yahoo.com/quote/BTC-USD/)
        """)


        # Educational Resources
        st.subheader("1. Introduction to Bitcoin")
        st.markdown("""
        - [Bitcoin Whitepaper - Satoshi Nakamoto](https://bitcoin.org/bitcoin.pdf)
        - [Investopedia: What is Bitcoin?](https://www.investopedia.com/terms/b/bitcoin.asp)
        """)
        st.subheader("2. Bitcoin Market Trends")
        st.markdown("""
        - [CoinMarketCap - Bitcoin Price and Market Cap](https://coinmarketcap.com/currencies/bitcoin/)
        - [TradingView - Bitcoin Charts](https://www.tradingview.com/symbols/BTCUSD/)
        """)
        st.subheader("3. Blockchain Technology Resources")
        st.markdown("""
        - [Blockchain Basics](https://www.ibm.com/topics/what-is-blockchain)
        - [Understanding Blockchain by Ethereum Foundation](https://ethereum.org/en/developers/docs/)
        """)



st.markdown("---")
st.markdown("Made with ❤️ by Mikaela | Powered by Machine Learning")
