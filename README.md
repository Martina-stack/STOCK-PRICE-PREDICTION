# Stock Price Prediction using LSTM

This project predicts the stock prices of a company using an LSTM (Long Short-Term Memory) model built with TensorFlow/Keras.

## Features
- Predicts stock 'Open', 'Close', 'High', 'Low' prices
- Uses historical stock data for training
- Visualizes predicted vs actual prices

## Files
- `checkprice` : UI and interface of the application
- `stock_price_lstm_model.keras` : Trained LSTM model
- `README.md` : Project description

## Technologies Used
- Python
- TensorFlow / Keras
- Google Colab
- Git & GitHub

## How to Use
1. Download the repository.
2. Load the model in Python:
```python
from tensorflow.keras.models import load_model
model = load_model('stock_price_lstm_model.keras')
