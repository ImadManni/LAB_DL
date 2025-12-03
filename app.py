import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf
from datetime import datetime, timedelta
import os

# Page config
st.set_page_config(
    page_title="AI Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">📈 AI Stock Price Predictor</h1>', unsafe_allow_html=True)
st.write("""
This application predicts stock prices using an LSTM neural network. 
Enter a stock symbol (e.g., AAPL, GOOGL, MSFT) and select a date range to see predictions.
""")

# Sidebar for user input
st.sidebar.header('User Input')
ticker = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')
days_to_predict = st.sidebar.slider('Days to Predict', 1, 30, 7)

# Model parameters
sequence_length = 60
model_path = 'stock_predictor.h5'

def create_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data(data, sequence_length=60):
    sc = MinMaxScaler(feature_range=(0, 1))
    scaled_data = sc.fit_transform(data)
    
    X = []
    y = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, sc

@st.cache_data
def load_data(ticker, period='2y'):
    try:
        df = yf.download(ticker, period=period, progress=False)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Main app
if st.sidebar.button('Predict'):
    with st.spinner('Fetching data and making predictions...'):
        # Load data
        df = load_data(ticker)
        
        if df is not None and not df.empty:
            # Display raw data
            st.subheader(f'Historical Data for {ticker}')
            st.dataframe(df.tail())
            
            # Prepare data for LSTM
            data = df['Close'].values.reshape(-1, 1)
            X, y, scaler = prepare_data(data, sequence_length)
            
            # Create and train model
            model = create_model()
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            
            # Make predictions
            last_sequence = data[-sequence_length:]
            last_sequence_scaled = scaler.transform(last_sequence)
            
            predictions = []
            for _ in range(days_to_predict):
                x_pred = last_sequence_scaled[-sequence_length:].reshape(1, sequence_length, 1)
                pred = model.predict(x_pred, verbose=0)
                predictions.append(pred[0][0])
                last_sequence_scaled = np.append(last_sequence_scaled, pred)
                
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = scaler.inverse_transform(predictions)
            
            # Create future dates
            last_date = df.index[-1]
            prediction_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
            
            # Plot results
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df.index[-60:], df['Close'].values[-60:], label='Historical Data')
            ax.plot(prediction_dates, predictions, 'r-', label='Predicted Prices')
            ax.set_title(f'{ticker} Stock Price Prediction')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
            # Show prediction table
            st.subheader('Price Predictions')
            pred_df = pd.DataFrame({
                'Date': prediction_dates,
                'Predicted Price': predictions.flatten()
            })
            st.dataframe(pred_df.set_index('Date').style.format({'Predicted Price': '${:,.2f}'}))
            
            # Save the model
            model.save(model_path)
            st.success(f'Model saved as {model_path}')

# Add some information about the app
st.sidebar.markdown("""
---
### How to Use
1. Enter a stock ticker symbol (e.g., AAPL, GOOGL, MSFT)
2. Select number of days to predict
3. Click 'Predict' to see the analysis

### About
This app uses an LSTM neural network to predict stock prices. 
The model is trained on historical price data to forecast future prices.
""")

# Add footer
st.markdown("""
---
*Built with ❤️ using Streamlit and TensorFlow*
""")