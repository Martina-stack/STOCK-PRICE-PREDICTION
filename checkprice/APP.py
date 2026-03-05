import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

# PAGE CONFIG
st.set_page_config(page_title="Stock Price LSTM Predictor", layout="wide")

# add custom CSS for look and feel
def _apply_custom_styles():
    css = """
    <style>
    /* page background gradient */
    .main .block-container {background: linear-gradient(135deg, #e8eef3 0%, #ffffff 100%); padding: 1.5rem 3rem;}
    /* professional font stack */
    h1,h2,h3,h4,h5, body {font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif;}
    h1 {color:#2c3e50; font-weight:700;}
    h2 {color:#34495e;}
    /* sidebar styling */
    .sidebar .sidebar-content {background-color:#1f2a36; color:#bdc3c7;}
    .sidebar .sidebar-content a, .sidebar .sidebar-content label {color:#bdc3c7;}
    /* buttons */
    button.stButton>button {background-color:#2c3e50; color:#ffffff; border-radius:4px; padding:0.6rem 1.2rem;}
    button.stButton>button:hover {background-color:#1a252f;}
    /* tables */
    table {border-collapse:collapse; width:100%; font-size:0.9rem;}
    table tr:nth-child(even){background:#f2f6f9;}
    table th{background:#2c3e50; color:#fff; padding:0.6rem;}
    table td{padding:0.6rem;}
    /* info card */
    .info-card {background:#ffffff; border-radius:8px; padding:1.2rem; box-shadow:0 3px 12px rgba(0,0,0,0.1); margin-bottom:1.2rem;}
    .info-card h2 {margin-top:0; color:#2c3e50;}
    /* forecast boxes */
    .forecast-box {background:#ffffff; border:1px solid #2c3e50; border-radius:8px; padding:1rem; font-family:'Roboto', sans-serif; margin-bottom:1rem;}
    .forecast-box h3 {margin-top:0; color:#2c3e50; font-style:italic;}
    /* hide footer */
    footer{visibility:hidden;}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
_apply_custom_styles()

# styled header
st.markdown("<h1 style='text-align:center;'>📈 <span style='color:#3498db;'>Stock Price Predictor</span></h1>", unsafe_allow_html=True)


# SIDEBAR CONFIG
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Symbol", "TSLA", help="Enter stock symbol (e.g., AAPL, TSLA, MSFT)")
start_date = st.sidebar.date_input(" Start Date", datetime(2014, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2026, 3, 1))
lookback_period = st.sidebar.slider(" Lookback Period (days)", 30, 60, 60)

run_analysis = st.sidebar.button("Run Analysis", key="run_btn")

# LOAD / RETRAIN LSTM MODEL
retrain = st.sidebar.checkbox("Retrain Model on Data", value=False)

units1 = st.sidebar.number_input("LSTM units layer 1", min_value=10, max_value=200, value=50, step=10)
units2 = st.sidebar.number_input("LSTM units layer 2", min_value=10, max_value=200, value=50, step=10)

dropout_rate = st.sidebar.slider("Dropout rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
epochs = st.sidebar.slider("Training epochs", min_value=10, max_value=200, value=50, step=10)
batch_size = st.sidebar.selectbox("Batch size", [16,32,64,128], index=1)

# choose model file path; automatically pick the '(1)' copy if it exists
alt_path = r"C:\Users\LENOV\Downloads\stock_price_lstm_model.keras"
default_path= r"C:\Users\LENOV\Downloads\stock_price_lstm_model (1).keras"
if os.path.exists(alt_path) and not os.path.exists(default_path):
    default_path = alt_path
model_path = st.sidebar.text_input("Model file path", default_path)

if retrain:
    lstm_model = None
    st.sidebar.info("Model will be retrained after data is downloaded")
else:
    try:
        lstm_model = load_model(model_path)
        st.sidebar.success(f"Pretrained model loaded from {model_path}")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        st.stop()

# MAIN ANALYSIS
if run_analysis:
    try:
        with st.spinner("Running analysis..."):
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
            
            if df.empty:
                st.error(f" No data found for {ticker}")
                st.stop()
            
            df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={'Adj Close': 'Adjusted Close'})
            
            # FEATURE ENG (no progress messages to keep UI clean)
        
        # Moving Averages
        df['SMA_10'] = df['Adjusted Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Adjusted Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Adjusted Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Adjusted Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        window = 14
        avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adjusted Close', 'SMA_10', 'SMA_20', 'SMA_50', 'RSI']
        df_model = df[features].copy().dropna()
        
        # SCALING
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_model)
        
        timestep = 60
        X = []
        y = []
        
        open_col_idx = df_model.columns.get_loc('Open')
        close_col_idx = df_model.columns.get_loc('Close')
        
        for i in range(timestep, len(scaled_data)):
            X.append(scaled_data[i-timestep:i])
            y.append([scaled_data[i, open_col_idx], scaled_data[i, close_col_idx]])
        
        X = np.array(X)
        y = np.array(y)
        
        # Train-test split
        split_ratio = 0.8
        split_index = int(len(X) * split_ratio)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        if retrain:
            # retraining message hidden under spinner
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout

            model = Sequential()
            model.add(LSTM(units=units1, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(dropout_rate))
            model.add(LSTM(units=units2, return_sequences=False))
            model.add(Dropout(dropout_rate))
            model.add(Dense(units=2, activation='linear'))

            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=0
            )
            lstm_model = model
            st.success("Retraining complete")

            if st.sidebar.checkbox("Save retrained model", value=False):
                save_path = st.sidebar.text_input("Model save path", model_path)
                lstm_model.save(save_path)
                st.sidebar.success(f"Model saved to {save_path}")

        # LSTM PREDICTIONS
        y_pred_scaled = lstm_model.predict(X_test, verbose=0)
        
        y_test_full_dim = np.zeros((len(y_test), scaled_data.shape[1]))
        y_test_full_dim[:, open_col_idx] = y_test[:, 0]
        y_test_full_dim[:, close_col_idx] = y_test[:, 1]
        y_test_unscaled = scaler.inverse_transform(y_test_full_dim)
        
        y_pred_full_dim = np.zeros((len(y_pred_scaled), scaled_data.shape[1]))
        y_pred_full_dim[:, open_col_idx] = y_pred_scaled[:, 0]
        y_pred_full_dim[:, close_col_idx] = y_pred_scaled[:, 1]
        y_pred_unscaled = scaler.inverse_transform(y_pred_full_dim)
        
        actual_open = y_test_unscaled[:, open_col_idx]
        actual_close = y_test_unscaled[:, close_col_idx]
        predicted_open = y_pred_unscaled[:, open_col_idx]
        predicted_close = y_pred_unscaled[:, close_col_idx]
        
        # METRICS
        st.header("Model Performance Metrics")
        
        mae_close = mean_absolute_error(actual_close, predicted_close)
        rmse_close = np.sqrt(mean_squared_error(actual_close, predicted_close))
        r2_close = r2_score(actual_close, predicted_close)
        
        mae_open = mean_absolute_error(actual_open, predicted_open)
        rmse_open = np.sqrt(mean_squared_error(actual_open, predicted_open))
        r2_open = r2_score(actual_open, predicted_open)
        
        avg_close_test = np.mean(actual_close)
        avg_open_test = np.mean(actual_open)
        
        # for comparison: mean of entire dataset (not just test subset)
        avg_close_all = np.mean(df_model['Close'])
        avg_open_all = np.mean(df_model['Open'])
        
        mae_close_pct_test = (mae_close / avg_close_test) * 100
        rmse_close_pct_test = (rmse_close / avg_close_test) * 100
        mae_open_pct_test = (mae_open / avg_open_test) * 100
        rmse_open_pct_test = (rmse_open / avg_open_test) * 100
        
        mae_close_pct_all = (mae_close / avg_close_all) * 100
        rmse_close_pct_all = (rmse_close / avg_close_all) * 100
        mae_open_pct_all = (mae_open / avg_open_all) * 100
        rmse_open_pct_all = (rmse_open / avg_open_all) * 100
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        col1.metric("Close MAE% (test)", f"{mae_close_pct_test:.2f}%")
        col2.metric("Close RMSE% (test)", f"{rmse_close_pct_test:.2f}%")
        col3.metric("Close R²", f"{r2_close:.4f}")
        col4.metric("Open MAE% (test)", f"{mae_open_pct_test:.2f}%")
        col5.metric("Open RMSE% (test)", f"{rmse_open_pct_test:.2f}%")
        col6.metric("Open R²", f"{r2_open:.4f}")
        
        st.write("---")
        
        # DETAILED METRICS TABLE
        st.subheader("Detailed Metrics Summary")
        metrics_data = {
            'Metric': ['MAE', 'MAE % (test)', 'MAE % (all)', 'RMSE', 'RMSE % (test)', 'RMSE % (all)', 'R² Score', 'Avg Price (test)', 'Avg Price (all)'],
            'Open Price': [
                f"{mae_open:.4f}",
                f"{mae_open_pct_test:.2f}%",
                f"{mae_open_pct_all:.2f}%",
                f"{rmse_open:.4f}",
                f"{rmse_open_pct_test:.2f}%",
                f"{rmse_open_pct_all:.2f}%",
                f"{r2_open:.4f}",
                f"{avg_open_test:.2f}",
                f"{avg_open_all:.2f}"
            ],
            'Close Price': [
                f"{mae_close:.4f}",
                f"{mae_close_pct_test:.2f}%",
                f"{mae_close_pct_all:.2f}%",
                f"{rmse_close:.4f}",
                f"{rmse_close_pct_test:.2f}%",
                f"{rmse_close_pct_all:.2f}%",
                f"{r2_close:.4f}",
                f"{avg_close_test:.2f}",
                f"{avg_close_all:.2f}"
            ]
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        
        # LAST N DAYS PREDICTIONS TABLE
        st.subheader(f"Last {lookback_period} Days Predictions")
        
        last_n_idx = min(lookback_period, len(actual_close))
        last_n_data = pd.DataFrame({
            'Day': range(1, last_n_idx + 1),
            'Actual Open': actual_open[-last_n_idx:].round(2),
            'Predicted Open': predicted_open[-last_n_idx:].round(2),
            'Open Diff': (predicted_open[-last_n_idx:] - actual_open[-last_n_idx:]).round(2),
            'Actual Close': actual_close[-last_n_idx:].round(2),
            'Predicted Close': predicted_close[-last_n_idx:].round(2),
            'Close Diff': (predicted_close[-last_n_idx:] - actual_close[-last_n_idx:]).round(2)
        })
        st.dataframe(last_n_data, use_container_width=True)
        
        # GRAPHS=
        st.subheader(" Price Predictions - Last 60 Days")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Open Price
        plot_range = min(60, len(actual_open))
        plot_index = range(plot_range)
        
        axes[0].plot(plot_index, actual_open[-plot_range:], label='Actual Open', color='blue', linewidth=2)
        axes[0].plot(plot_index, predicted_open[-plot_range:], label='Predicted Open', color='red', linestyle='--', linewidth=2)
        axes[0].set_title('Open Price Prediction vs. Actual (Last 60 Days)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Days')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Close Price
        axes[1].plot(plot_index, actual_close[-plot_range:], label='Actual Close', color='green', linewidth=2)
        axes[1].plot(plot_index, predicted_close[-plot_range:], label='Predicted Close', color='orange', linestyle='--', linewidth=2)
        axes[1].set_title('Close Price Prediction vs. Actual (Last 60 Days)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Days')
        axes[1].set_ylabel('Price ($)')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # NEXT DAY FORECAST
        st.header("Next Trading Day Forecast")
        
        last_timestep_data = df_model.tail(timestep)
        scaled_last_timestep_data = scaler.transform(last_timestep_data)
        X_next_day = scaled_last_timestep_data.reshape(1, timestep, scaled_last_timestep_data.shape[1])
        
        predicted_next_day_scaled = lstm_model.predict(X_next_day, verbose=0)
        
        predicted_next_day_full_dim = np.zeros((1, scaled_data.shape[1]))
        predicted_next_day_full_dim[0, open_col_idx] = predicted_next_day_scaled[0, 0]
        predicted_next_day_full_dim[0, close_col_idx] = predicted_next_day_scaled[0, 1]
        
        predicted_next_day_unscaled = scaler.inverse_transform(predicted_next_day_full_dim)
        
        forecasted_open = predicted_next_day_unscaled[0, open_col_idx]
        forecasted_close = predicted_next_day_unscaled[0, close_col_idx]
        
        last_actual_close = df_model['Close'].iloc[-1]
        
        open_change = forecasted_open - last_actual_close
        close_change = forecasted_close - last_actual_close
        
        open_pct_return = (open_change / last_actual_close) * 100
        close_pct_return = (close_change / last_actual_close) * 100
        
        open_direction = ' UP' if open_change > 0 else ' DOWN' if open_change < 0 else '➡️ NO CHANGE'
        close_direction = ' UP' if close_change > 0 else ' DOWN' if close_change < 0 else '➡️ NO CHANGE'
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='forecast-box'>", unsafe_allow_html=True)
            st.subheader("Open Price")
            st.metric("Forecasted Price", f"${forecasted_open:.2f}")
            st.write(f"Change from Last Close: **{open_change:.2f}** ({open_pct_return:+.2f}%)")
            st.write(f"Direction: {open_direction}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='forecast-box'>", unsafe_allow_html=True)
            st.subheader("Close Price")
            st.metric("Forecasted Price", f"${forecasted_close:.2f}")
            st.write(f"Change from Last Close: **{close_change:.2f}** ({close_pct_return:+.2f}%)")
            st.write(f"Direction: {close_direction}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.write("---")
        
        st.success("Analysis completed successfully!")
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        import traceback
        st.write(traceback.format_exc())
else:
    st.info("Configure parameters in the sidebar and click 'Run Analysis' to start")
    # about section removed for a cleaner idle view
