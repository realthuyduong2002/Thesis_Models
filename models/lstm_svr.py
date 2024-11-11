import numpy as np
import sys
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, f1_score
from sklearn.svm import SVR
import math
from datetime import timedelta

# Thêm đường dẫn thư mục cha để có thể import stock_data_processing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import các biến và hàm từ stock_data_processing.py
from stock_data_processing import fetch_and_save_stock_data, check_missing_symbols, folder_path, symbols

# Đọc dữ liệu từ CSV và chuẩn bị
def load_stock_data(symbol):
    file_path = os.path.join(folder_path, f'{symbol}.csv')
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df[['Date', 'Close']]
    return df

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

for symbol in symbols:
    print(f"Processing {symbol}...")
    df = load_stock_data(symbol)
    x_data, y_data, scaler = prepare_data(df)

    # Chia dữ liệu train/test
    train_size = int(len(x_data) * 0.8)
    x_train, x_test = x_data[:train_size], x_data[train_size:]
    y_train, y_test = y_data[:train_size], y_data[train_size:]

    # Xây dựng và huấn luyện mô hình LSTM
    lstm_model = build_lstm_model((x_train.shape[1], 1))
    lstm_model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=0)

    # Dự đoán với LSTM
    lstm_predictions = lstm_model.predict(x_test)
    lstm_predictions_rescaled = scaler.inverse_transform(lstm_predictions)

    # Chuyển đổi y_test về giá trị gốc
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Tính phần dư (residuals) từ dự đoán LSTM
    residuals = y_test_rescaled - lstm_predictions_rescaled

    # Chuẩn bị dữ liệu cho mô hình SVR
    x_train_flat = x_train[:, :, 0]
    x_test_flat = x_test[:, :, 0]

    if len(x_test_flat) != len(residuals):
        residuals = residuals[:len(x_test_flat)]

    # Dự đoán phần dư với SVR
    svr_model = SVR(kernel='rbf')
    svr_model.fit(x_train_flat, y_train)
    svr_residual_pred = svr_model.predict(x_test_flat).reshape(-1, 1)

    # Kết hợp dự đoán của LSTM và SVR
    final_predictions = lstm_predictions_rescaled + svr_residual_pred

    # Tính các chỉ số đánh giá
    mse = mean_squared_error(y_test_rescaled, final_predictions)
    rmse = math.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test_rescaled, final_predictions)
    mae = np.mean(np.abs(y_test_rescaled - final_predictions))

    # Tính F1-score bằng cách phân loại tăng/giảm
    actual_direction = (y_test_rescaled[1:] > y_test_rescaled[:-1]).astype(int).flatten()
    predicted_direction = (final_predictions[1:] > final_predictions[:-1]).astype(int).flatten()
    f1 = f1_score(actual_direction, predicted_direction)

    print(f"RMSE for {symbol}: {rmse}")
    print(f"MSE for {symbol}: {mse}")
    print(f"MAPE for {symbol}: {mape}")
    print(f"MAE for {symbol}: {mae}")
    print(f"F1 Score for {symbol}: {f1}")

    # Dự báo 1 năm tiếp theo
    future_prices = []
    last_sequence = x_test[-1]
    current_date = df['Date'].max()

    for i in range(365):
        lstm_next_pred = lstm_model.predict(last_sequence.reshape(1, -1, 1))
        svr_next_residual = svr_model.predict(last_sequence[:, 0].reshape(1, -1))
        next_price = lstm_next_pred + svr_next_residual
        next_price_rescaled = scaler.inverse_transform(next_price.reshape(-1, 1))
        future_prices.append(next_price_rescaled[0, 0])
        last_sequence = np.append(last_sequence[1:], lstm_next_pred, axis=0)
        current_date += timedelta(days=1)

    # Lưu kết quả
    forecast_results.append({
        'Symbol': symbol,
        'RMSE': rmse,
        'MSE': mse,
        'MAPE': mape,
        'MAE': mae,
        'F1 Score': f1,
        'Predicted_Prices': final_predictions.flatten().tolist(),
        'Actual_Prices': y_test_rescaled.flatten().tolist(),
        'Future_Price_Predictions': future_prices
    })

# Tạo DataFrame từ danh sách kết quả và lưu vào CSV
summary_hybrid = pd.DataFrame(forecast_results)
summary_hybrid['Predicted_Prices'] = summary_hybrid['Predicted_Prices'].apply(lambda x: ', '.join(map(str, x)))
summary_hybrid['Actual_Prices'] = summary_hybrid['Actual_Prices'].apply(lambda x: ', '.join(map(str, x)))
summary_hybrid['Future_Price_Predictions'] = summary_hybrid['Future_Price_Predictions'].apply(lambda x: ', '.join(map(str, x)))

# Lưu DataFrame vào file CSV
output_file = os.path.join(os.path.dirname(__file__), 'forecast_summary_hybrid.csv')
summary_hybrid.to_csv(output_file, index=False)