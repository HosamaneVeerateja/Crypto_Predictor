import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import streamlit as st

# Function to load data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Check if there's enough data (at least 10 days)
    if len(data) < 10:
        raise ValueError("Not enough data for prediction. Please select a longer date range.")
    
    return data['Close']

# Function to preprocess data
def preprocess_data(data, time_step=10):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_data) - time_step):
        X.append(scaled_data[i:i + time_step, 0])
        y.append(scaled_data[i + time_step, 0])

    return np.array(X), np.array(y), scaler

# Function to train the Random Forest model
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

# Function to predict future prices
def predict_future(model, last_data, scaler, days_to_predict=5):
    future_predictions = []
    current_input = last_data
    
    for _ in range(days_to_predict):
        # Predict the next day's price
        prediction = model.predict(current_input.reshape(1, -1))
        future_predictions.append(prediction[0])
        
        # Prepare the input for the next prediction
        current_input = np.append(current_input[1:], prediction)
    
    # Inverse transform the predictions to actual values
    future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    return future_predictions_rescaled

# Streamlit UI
st.title("Crypto Price Predictor using Random Forest")

# User inputs
ticker = st.text_input("Enter Cryptocurrency Symbol (e.g., BTC-USD)", "BTC-USD")
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))
days_to_predict = st.slider("Number of Days to Predict", 1, 30, 5)

if st.button("Predict"):
    # Load the data
    data = load_data(ticker, start_date, end_date)
    st.line_chart(data)

    # Preprocess the data
    time_step = 10
    X, y, scaler = preprocess_data(data, time_step)

    # Train the model
    model = train_model(X, y)

    # Make predictions for the available data
    predictions = model.predict(X)
    predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))

    # Plot predictions vs actual data
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[time_step:], data[time_step:], label="Actual Prices", color="blue")
    plt.plot(data.index[time_step:], predictions_rescaled, label="Predicted Prices", color="red", linestyle="--")
    plt.legend()
    plt.title(f"{ticker} Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    st.pyplot(plt)

    # Predict future prices
    last_data = X[-1]
    future_predictions = predict_future(model, last_data, scaler, days_to_predict)

    # Display future predictions
    future_dates = pd.date_range(data.index[-1], periods=days_to_predict + 1, freq='D')[1:]
    st.subheader(f"Predicted Future Prices for {ticker}")
    future_df = pd.DataFrame(future_predictions, columns=["Predicted Price"], index=future_dates)
    st.write(future_df)

    # Display error metric
    mae = mean_absolute_error(data[time_step:], predictions_rescaled)
    st.write(f"Mean Absolute Error: {mae:.2f}")
