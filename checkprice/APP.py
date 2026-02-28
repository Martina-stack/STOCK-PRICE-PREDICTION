import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor App")

# Sidebar: User inputs
st.sidebar.header("User Input")
stock = st.sidebar.text_input("Ticker symbol", "NVDA")

default_start = pd.to_datetime("2021-02-01")
default_end = pd.to_datetime("2026-02-27")
start_date = st.sidebar.date_input("Start date", default_start)
end_date = st.sidebar.date_input("End date", default_end)

model = load_model(r"C:\\Users\\LENOV\\Downloads\\stock_model_lstm.h5")

with st.spinner("Fetching Stock Data..."):
    data = yf.download(stock, start=start_date, end=end_date)

st.subheader("Raw Stock Data")
st.dataframe(data, use_container_width=True)

data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

scaler = MinMaxScaler(feature_range=(0,1))
# Prepare last 60 days data
close_prices = data[['Close']].copy()
scaled_close = scaler.fit_transform(close_prices)

sequence_length = 60
X = []
for i in range(sequence_length, len(scaled_close)):
    X.append(scaled_close[i-sequence_length:i, 0])

X = np.array(X)
X = X.reshape(X.shape[0], X.shape[1], 1)

last_60_X = X[-60:]
last_60_real = data['Close'].tail(60)
last_60_dates = last_60_real.index

# Predict last 60 days
predicted_last_60 = model.predict(last_60_X)
predicted_last_60 = scaler.inverse_transform(predicted_last_60)

# Predict tomorrow
latest_sequence = scaled_close[-60:].reshape(1,60,1)
scaled_tomorrow = model.predict(latest_sequence)
tomorrow_price = float(scaler.inverse_transform(scaled_tomorrow)[0][0])
today_price = float(data['Close'].iloc[-1])
price_change = tomorrow_price - today_price
return_percent = (price_change / today_price) * 100
direction = "UP 📈" if price_change > 0 else "DOWN 📉"

# Plot last 60 days + tomorrow prediction
st.subheader("Last 60 Days: Actual vs Model Prediction")
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(last_60_dates, last_60_real.values, label="Actual Close Price", color='blue')
ax.plot(last_60_dates, predicted_last_60.flatten(), label="Model Predicted", color='orange')
ax.scatter(pd.to_datetime(data.index[-1]) + pd.Timedelta(days=1), tomorrow_price, color='red', label="Tomorrow Prediction")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title(f"{stock} Stock Price Prediction")
ax.legend()
st.pyplot(fig)
# Show metrics at the bottom
st.subheader("Prediction Metrics")
st.write(f"Today's Close: ${today_price:.2f}")
st.write(f"Tomorrow's Predicted Close: ${tomorrow_price:.2f}")
st.write(f"Price Change: ${price_change:.2f}")
st.write(f"Return (%): {return_percent:.2f}%")
st.write(f"Direction: {direction}")