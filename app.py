
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

model = load_model('Stock_Predictions_Model.keras')

st.set_page_config(page_title="LSTM Stock Price Predictor", layout="wide")
st.title("📈 Bi-LSTM Stock Price Predictor")
st.markdown("Predict future stock prices using Bi-directional LSTM Neural Network.")

# Sidebar for input
stock = st.sidebar.text_input("Enter Stock Ticker (e.g., GOOG, AAPL, TSLA):", value="GOOG")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2012-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-04-05"))

if st.sidebar.button("Predict"):
    data = yf.download(stock, start=start_date, end=end_date)
    data.dropna(inplace=True)
    data.reset_index(inplace=True)

    st.subheader("📊 Raw Stock Data")
    st.dataframe(data.tail(10))

    data['50_MA'] = data['Close'].rolling(50).mean()
    data['100_MA'] = data['Close'].rolling(100).mean()
    data['200_MA'] = data['Close'].rolling(200).mean()

    st.subheader("📉 Price with Moving Averages")
    fig_ma, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Close'], label='Close', color='blue')
    ax.plot(data['50_MA'], label='50 MA', color='orange')
    ax.plot(data['100_MA'], label='100 MA', color='green')
    ax.plot(data['200_MA'], label='200 MA', color='purple')
    ax.set_title(f"{stock} Stock Price with Moving Averages")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig_ma)

    close_data = data[['Date', 'Close']]
    train_size = int(len(close_data) * 0.6)
    data_train = close_data[:train_size].copy()
    data_test = close_data[train_size:].copy()

    scaler = MinMaxScaler()
    data_train_scaled = scaler.fit_transform(data_train[['Close']])

    x_train, y_train = [], []
    for i in range(100, len(data_train_scaled)):
        x_train.append(data_train_scaled[i - 100:i])
        y_train.append(data_train_scaled[i])

    past_100 = data_train.tail(100)
    final_test_data = pd.concat([past_100, data_test], ignore_index=True)
    final_test_scaled = scaler.transform(final_test_data[['Close']])

    x_test, y_test = [], []
    for i in range(100, len(final_test_scaled)):
        x_test.append(final_test_scaled[i - 100:i])
        y_test.append(final_test_scaled[i])

    x_test, y_test = np.array(x_test), np.array(y_test)

    y_predicted = model.predict(x_test)
    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    st.subheader("📊 Actual vs Predicted Stock Price")
    fig_pred, ax2 = plt.subplots(figsize=(14, 6))
    ax2.plot(y_test, label='Actual', color='blue')
    ax2.plot(y_predicted, label='Predicted', color='red')
    ax2.fill_between(range(len(y_test)), y_test.ravel(), y_predicted.ravel(), color='gray', alpha=0.2, label='Prediction Gap')
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Stock Price")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig_pred)

    min_len = min(len(data_test['Date'].iloc[100:]), len(y_test), len(y_predicted))

    results_df = pd.DataFrame({
        'Date': data_test['Date'].iloc[100:100+min_len].values,
        'Actual Price': y_test.ravel()[:min_len],
        'Predicted Price': y_predicted.ravel()[:min_len]
    })

    st.subheader("🧾 Actual vs Predicted Table")
    st.dataframe(results_df.head(20))

    st.download_button("📥 Download CSV", data=results_df.to_csv(index=False), file_name="actual_vs_predicted.csv")
