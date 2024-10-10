import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from flask import Flask, render_template, request
import io
import base64

app = Flask(__name__)

# Function to fetch cryptocurrency data
def get_crypto_data(crypto_id, currency, days):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {'vs_currency': currency, 'days': days}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code}")
    data = response.json()
    if 'prices' not in data:
        raise KeyError("'prices' not found in API response")
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Function to get top 20 cryptocurrencies
def get_top_20_cryptos():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 20, 'page': 1}
    response = requests.get(url, params=params)
    data = response.json()
    top_20 = [{'id': coin['id'], 'symbol': coin['symbol'].upper(), 'name': coin['name']} for coin in data]
    return top_20

# Function to create plots and predictions
def create_plots(crypto_id):
    # Load data
    df = get_crypto_data(crypto_id, 'usd', 365)
    df = df.sort_values('date')  # Ensure data is sorted

    # -------------------------
    # Linear Regression
    # -------------------------

    # Prepare data
    X_lr = np.array(range(len(df))).reshape(-1, 1)  # Days as numbers
    y_lr = df['price'].values

    # Train model on all available data
    model_lr = LinearRegression()
    model_lr.fit(X_lr, y_lr)

    # Predict future prices
    future_days = 10
    total_days = len(df) + future_days
    X_future_lr = np.array(range(len(df), total_days)).reshape(-1, 1)
    y_pred_future_lr = model_lr.predict(X_future_lr)

    # Generate future dates
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days)

    # Plot Linear Regression
    fig1 = plt.figure(figsize=(12,6))
    plt.plot(df['date'], df['price'], label='Actual Prices')
    plt.plot(future_dates, y_pred_future_lr, label='Predicted Prices (Linear Regression)', color='red')
    plt.title(f'Predicted Prices for {crypto_id.capitalize()} (Linear Regression)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot to string in base64 format
    img1 = io.BytesIO()
    fig1.savefig(img1, format='png')
    img1.seek(0)
    plot_url1 = base64.b64encode(img1.getvalue()).decode()

    plt.close(fig1)

    # -------------------------
    # LSTM Model
    # -------------------------

    # Prepare data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df[['price']].values)

    sequence_length = 60

    def create_sequences(data, seq_length):
        X = []
        y = []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    X_lstm, y_lstm = create_sequences(scaled_data, sequence_length)

    # Train model on all available data
    model_lstm = Sequential()
    model_lstm.add(LSTM(units=100, return_sequences=True, input_shape=(sequence_length, 1)))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(units=100, return_sequences=False))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(units=50))
    model_lstm.add(Dense(units=1))

    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    model_lstm.fit(X_lstm, y_lstm, batch_size=32, epochs=10, verbose=0)

    # Predict future prices
    future_predictions = []
    last_sequence = scaled_data[-sequence_length:]

    for _ in range(future_days):
        prediction = model_lstm.predict(last_sequence.reshape(1, sequence_length, 1))
        future_predictions.append(prediction[0][0])
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)

    future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))

    # Plot LSTM
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(df['date'], df['price'], label='Actual Prices')
    plt.plot(future_dates, future_predictions_rescaled, label='Predicted Prices (LSTM)', color='red')
    plt.title(f'Predicted Prices for {crypto_id.capitalize()} (LSTM)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    img2 = io.BytesIO()
    fig2.savefig(img2, format='png')
    img2.seek(0)
    plot_url2 = base64.b64encode(img2.getvalue()).decode()

    plt.close(fig2)

    return plot_url1, plot_url2

@app.route('/', methods=['GET', 'POST'])
def home():
    top_20 = get_top_20_cryptos()
    selected_crypto = 'bitcoin'
    if request.method == 'POST':
        selected_crypto = request.form.get('crypto_select')
    try:
        plot_url1, plot_url2 = create_plots(selected_crypto)
        return render_template('index.html', plot_url1=plot_url1, plot_url2=plot_url2,
                               top_20=top_20, selected_crypto=selected_crypto)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
