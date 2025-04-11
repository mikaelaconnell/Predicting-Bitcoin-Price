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

# Set the streamlit app layout to wide
st.set_page_config(layout="wide")

st.title("Bitcoin Price Prediction App 💰")

# --- CSS Styling ---
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">

    <style>
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #121212;
        color: white;
    }

    /* Metric container */
    div[data-testid="stMetric"] {
        background-color: #1e1e1e;
        border-radius: 12px;
        padding: 1.2em;
        margin: 1.5em 0;
        box-shadow: 0 0 10px rgba(255,255,255,0.05);
    }

    /* Label */
    div[data-testid="stMetric"] label {
        color: #bbbbbb !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }

    /* Main Value */
    div[data-testid="stMetric"] div:nth-child(1) {
        color: #ffffff !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        text-shadow: 0 0 8px rgba(255, 255, 255, 0.2);
    }

    /* Delta Value */
    div[data-testid="stMetric"] div:nth-child(2) {
        font-weight: 600 !important;
    }

    /* Green glow for positive delta */
    div[data-testid="stMetric"] div:has(svg[data-testid="icon-arrow-up"]) {
        color: #00ff8b !important;
        text-shadow: 0 0 6px rgba(0, 255, 139, 0.6);
    }

    /* Red glow for negative delta */
    div[data-testid="stMetric"] div:has(svg[data-testid="icon-arrow-down"]) {
        color: #ff4d4d !important;
        text-shadow: 0 0 6px rgba(255, 77, 77, 0.6);
    }
    
    /* Orange Run Prediction Button */
    .stButton>button {
        background-color: #F7931A;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75em 1em;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: 0 0 8px rgba(247, 147, 26, 0.6);
        transition: all 0.3s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #ffa733;
        box-shadow: 0 0 12px rgba(247, 147, 26, 0.9);
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Predictive Analytics", "Data Exploration", "Forecast Customization", "Model Performance", "Data Sources"])   
tab_dict = {
    "Overview": tab1,
    "Predictive Analytics": tab2,
    "Data Exploration": tab3,
    "Forecast Customization": tab4,
    "Model Performance": tab5,
    "Data Sources": tab6
}


with tab_dict["Overview"]:
    
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
        fig = px.line(df, x='timestamp', y='price', title="Live Bitcoin Price (Last 30 Days)")

        # Display the plot as an interactive graph
        st.plotly_chart(fig)


        # --- Page Layout ---
        left_col, right_col = st.columns(2)
        with left_col:
            st.markdown("""
                ### About This App
                This Bitcoin Price Prediction App is a dynamic forecasting tool powered by machine learning and macroeconomic analysis.
                Designed for analysts, crytpto enthusiasts, and curious investors, the app blends real-time market data with historical 
                economic indicators to deliver meaningful insights into Bitcoin's price movements.
                    
                - **Real-time Bitcoin Price Tracking:** Visualize live BTC price movements alongside economic factors
                - **Predictive Analytics:** Leverage machine learning to forecast Bitcoin prices
                - **Data Exploration:** Explore the relationship between Bitcoin prices and key economic indicators through interactive charts
                - **Customizable Forecast Assumptions:** Tailor the model's assumptions to fit your investment strategy
                - **Model Performance:** Understand the model's performance with metrics like RMSE and R² score
                - **Data Sources:** Comprehensive data from reliable sources like CoinGecko, Yahoo Finance, and the Federal Reserve 
            """)

        with right_col:
            cg = CoinGeckoAPI()
            btc_data = cg.get_price(ids='bitcoin', vs_currencies='usd')
            latest_price = btc_data['bitcoin']['usd']
            st.metric(label="Latest BTC Price", value=f"${latest_price:,.2f}")
            st.image("image/bitcoin4.jpg", use_container_width=True) 
    

        # --- Additional Information ---
        st.write("""
            ### Key Features for Prediction
            - **Global M2 Money Supply:** Total amount of money circulating in the global economy
            - **U.S. M2 Money Supply:** Indicator of domestic economic conditions and liquidity
            - **Inflation Rate (CPI, PPI):** Impact on purchasing power and overall economic stability
            - **Interest Rates (Federal Funds Rate):** Influences demand for investment and speculative assets like Bitcoin
            - **Stock Market Indices (S&P 500, Nasdaq):** Reflects overall market sentiment and investor behavior
            - **Bitcoin Trading Volume & Dominance:** Measures the market activity and dominance of Bitcoin over other cryptocurrencies
            """)
                
        st.write("""
            ### How This Data Contributes to Bitcoin Price Prediction
            By combining multiple economic indicators like the M2 Money Supply, inflation rates, and stock market movements, this dataset allows us to capture the broader macroeconomic environment, which is crucial for predicting Bitcoin price fluctuations.
            Cryptocurrency markets are deeply influenced by broader economic trends. This app bridges the gap between crypto and macroeconomics
            by incorporating key financial indicators into the prediction model, providing more informed insights than purely technical approaches.
            """)

with tab_dict["Predictive Analytics"]:

        # --- Sidebar ---
        st.sidebar.title("Choose the prediction range and time interval below:")
        st.sidebar.write("After running the prediction, check the Predictive Analytics tab for the results!")
        st.sidebar.image("image/bitcoin_image3.jpg", use_container_width=True) 
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
            prediction_df = make_prediction(data, days)
            # Update metric
            predicted_price = float(prediction_df['predicted_price'].values[-1])
            st.metric(label="Predicted Price", value=f"${predicted_price:,.2f}")

            st.subheader("Predicted Bitcoin Prices 📈")

            # Plot predictions
            st.line_chart(prediction_df['predicted_price'])
    
            st.write("Note: The predictions are based on historical data and may not reflect actual future prices. The model uses XGBoost for predictions, and is trained on historical prices and macroeconomic indicators.")

            # --- Highlight Metrics ---
            st.subheader("Market Insights")
            st.metric(label="Latest BTC Price", value=f"${latest_price:,.2f}")

            price_change = predicted_price - latest_price
            trend = "Uptrend 📈 " if price_change > 0 else "Downtrend 📉"

            st.metric(label="Price Change", value=f"${price_change:,.2f}", delta=f"{price_change / latest_price * 100:.2f}%")
            st.metric(label="BTC Trend", value=trend)
            st.metric(label="Predicted Price", value=f"${predicted_price:,.2f}")
            st.metric(label="Buy/Sell Suggestion", value="Buy" if price_change > 0 else "Sell")
            st.metric(label="Take Profit", value=f"${predicted_price * 1.05:,.2f}")
            st.metric(label="Stop Loss", value=f"${predicted_price * 0.95:,.2f}")

with tab_dict["Data Exploration"]:
        st.header("Explore the Data")
        st.write("This section allows you to explore the relationship between Bitcoin prices and key economic indicators through interactive charts.")

        # Path to your CSV file
        csv_file_path = 'bitcoin_final_filled.csv'

        # Load the CSV file
        data = pd.read_csv(csv_file_path)

        # Ensure that the 'Date' column is in datetime format
        data['Date'] = pd.to_datetime(data['date'])


        # Extract Bitcoin price and Global M2 data
        bitcoin_price = data['btc_price']
        global_m2 = data['global_m2']


        # --- Interactive Visualizations ---
        # Dropdown to select feature for focus
        feature = st.selectbox('### Select feature to compare to Bitcoin Price:', ['global_m2', 'sp500', 'nasdaq', 'btc_volume'])

        # Create the figure with two Y axes
        fig3 = go.Figure()

        # Add Bitcoin price trace (left y-axis)
        fig3.add_trace(go.Scatter(x=data['date'], y=data['btc_price'], mode='lines', name='Bitcoin Price', 
                                line=dict(color='blue')))

        # Add selected feature trace (right y-axis)
        fig3.add_trace(go.Scatter(x=data['date'], y=data[feature], mode='lines', name=feature.capitalize(), 
                            line=dict(color='red'), yaxis="y2"))

        # Update layout for two Y axes
        fig3.update_layout(
            title=f"{feature.capitalize()} vs. Bitcoin Price (2015-2025)",
            xaxis_title="Date",
            yaxis_title="Bitcoin Price (USD)",
            yaxis2=dict(
                title=feature.capitalize(),
                overlaying="y",  # Share the x-axis but have a different scale for y2
                side="right"
            ),
            template="plotly_dark",  
            hovermode="x unified"  # Show hover data for both lines at the same time
        )

        # Display the plotly chart in Streamlit
        st.plotly_chart(fig3)


        # Display Correlation Matrix
        st.subheader("Correlation Matrix")
        # Create a correlation matrix for the selected columns
        corr_df = data[['btc_price', 'global_m2', 'us_m2_scaled', 'sp500', 'nasdaq', 'fed_funds_rate', 'cpi', 'ppi', 'btc_volume']]
        corr_matrix = corr_df.corr()
        corr_matrix = corr_matrix.round(2)

        # Create a Plotly heatmap for correlation matrix
        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.columns.tolist(),
            colorscale='Blues',
            font_colors=['white'] * len(corr_matrix.columns),
            showscale=True,
            colorbar=dict(title='Correlation', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
        )

        # Add interactivity: allow zoom and hover for more details
        fig.update_layout(
            xaxis_title="Features",
            yaxis_title="Features",
            template="plotly_dark",
            xaxis=dict(showgrid=True, showline=True),
            yaxis=dict(showgrid=True, showline=True),
            hovermode="closest",  # Enables hover-over features
            autosize=True
        )

        # Add a feature to select or deselect correlation columns
        selected_features = st.multiselect(
            "Select Features to Display in the Correlation Matrix:",
            options=corr_df.columns.tolist(),
            default=corr_df.columns.tolist()  # Show all by default
        )

        # Filter the correlation matrix based on selected features
        filtered_corr_matrix = corr_matrix[selected_features].loc[selected_features]

        # Create a new interactive heatmap with the filtered correlation matrix
        filtered_fig = ff.create_annotated_heatmap(
            z=filtered_corr_matrix.values,
            x=filtered_corr_matrix.columns.tolist(),
            y=filtered_corr_matrix.columns.tolist(),
            colorscale='Blues',
            font_colors=['white'] * len(filtered_corr_matrix.columns),
            showscale=True,
            colorbar=dict(title='Correlation', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
        )

        # Update layout for zoom functionality and interactivity
        filtered_fig.update_layout(
            title="Filtered Correlation Matrix (Interactive)",
            xaxis_title="Features",
            yaxis_title="Features",
            template="plotly_dark",
            xaxis=dict(showgrid=True, showline=True),
            yaxis=dict(showgrid=True, showline=True),
            hovermode="closest",  # Enables hover-over features
            autosize=True
        )

        # Display the correlation heatmap in Streamlit
        st.plotly_chart(filtered_fig)


with tab_dict["Forecast Customization"]:
        st.header("Customize Forecast Assumptions")
        inflation = st.slider("Expected Inflation Rate (%)", 0.0, 10.0, 2.5)
        m2_growth = st.slider("Global M2 Growth (%)", 0.0, 20.0, 5.0)
        st.write("Future forecast customization coming soon...")

with tab_dict["Model Performance"]:
        st.header("Model Insights")
        st.write("Coming soon: RMSE, R² score, residual analysis and predicted vs actual plots.")

with tab_dict["Data Sources"]:
        st.header("Data Sources & Educational Resources")
        st.write("""
        - **Global M2 Money Supply:** [World Bank](https://data.worldbank.org/indicator/FM.LBL.MQMY.CN)
        - **U.S. M2 Money Supply:** [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/series/M2SL)
        - **Inflation Rate (CPI, PPI):** [U.S. Bureau of Labor Statistics](https://www.bls.gov/cpi/)
        - **Interest Rates (Federal Funds Rate):** [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/series/FEDFUNDS)
        - **Stock Market Indices (S&P 500, Nasdaq):** [Yahoo Finance](https://finance.yahoo.com/)
        - **Bitcoin Trading Volume & Dominance:** [CoinMarketCap](https://coinmarketcap.com/)
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
st.markdown("Made with ❤️ by Mikaela | Powered by XGBoost + Streamlit")
