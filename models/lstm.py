import numpy as np
import sys
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, f1_score
import math
from datetime import timedelta

# Thêm đường dẫn thư mục cha để có thể import stock_data_processing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import các biến và hàm từ stock_data_processing.py
from stock_data_processing import folder_path, symbols

# Đọc dữ liệu từ CSV và chuẩn bị
def load_stock_data(symbol):
    file_path = os.path.join(folder_path, f'{symbol}.csv')
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df[['Date', 'Close']]
    return df

# Chuẩn bị dữ liệu cho một mã cổ phiếu
def prepare_data(df, time_step=60):
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x_data, y_data = [], []
    for i in range(time_step, len(scaled_data)):
        x_data.append(scaled_data[i-time_step:i, 0])
        y_data.append(scaled_data[i, 0])

    x_data, y_data = np.array(x_data), np.array(y_data)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    return x_data, y_data, scaler

# Xây dựng mô hình LSTM
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Tạo danh sách lưu kết quả
forecast_results = []

# Lặp qua từng mã cổ phiếu
for symbol in symbols:
    print(f"Processing {symbol}...")
    df = load_stock_data(symbol)
    x_data, y_data, scaler = prepare_data(df)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=False)

    model = build_lstm_model(x_train.shape)
    model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=0)

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Tính các chỉ số đánh giá
    mse = mean_squared_error(y_test_original, predicted_prices)
    rmse = math.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test_original, predicted_prices)
    mae = np.mean(np.abs(y_test_original - predicted_prices))

    # Tính F1-score bằng cách phân loại tăng/giảm
    actual_direction = (y_test_original[1:] > y_test_original[:-1]).astype(int).flatten()
    predicted_direction = (predicted_prices[1:] > predicted_prices[:-1]).astype(int).flatten()
    f1 = f1_score(actual_direction, predicted_direction)

    # Dự báo 1 năm tiếp theo
    future_prices = []
    last_sequence = x_test[-1]
    for i in range(365):
        next_price = model.predict(last_sequence.reshape(1, -1, 1))
        next_price_scaled = scaler.inverse_transform(next_price.reshape(-1, 1))
        future_prices.append(next_price_scaled[0, 0])
        last_sequence = np.append(last_sequence[1:], next_price, axis=0)

    # Lưu kết quả
    forecast_results.append({
        'Symbol': symbol,
        'RMSE': rmse,
        'MSE': mse,
        'MAPE': mape,
        'MAE': mae,
        'F1 Score': f1,
        'Predicted_Prices': predicted_prices.flatten().tolist(),
        'Actual_Prices': y_test_original.flatten().tolist(),
        'Future_Price_Predictions': future_prices
    })

# Tạo DataFrame từ kết quả và lưu vào CSV
summary_lstm = pd.DataFrame(forecast_results)
summary_lstm['Predicted_Prices'] = summary_lstm['Predicted_Prices'].apply(lambda x: ', '.join(map(str, x)))
summary_lstm['Actual_Prices'] = summary_lstm['Actual_Prices'].apply(lambda x: ', '.join(map(str, x)))
summary_lstm['Future_Price_Predictions'] = summary_lstm['Future_Price_Predictions'].apply(lambda x: ', '.join(map(str, x)))

# Lưu DataFrame vào file CSV
output_file = os.path.join(os.path.dirname(__file__), 'forecast_summary_lstm.csv')
summary_lstm.to_csv(output_file, index=False)