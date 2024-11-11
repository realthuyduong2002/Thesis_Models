import numpy as np
import os
import sys
import pandas as pd
import math
from sklearn.svm import SVR
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, f1_score

# Thêm đường dẫn thư mục cha để có thể import stock_data_processing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import các biến và hàm từ stock_data_processing.py
from stock_data_processing import folder_path, symbols

# Hàm load dữ liệu
def load_stock_data(symbol):
    file_path = os.path.join(folder_path, f'{symbol}.csv')
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df[['Date', 'Close']]  # Giữ lại cột Date và Close
    return df

# Hàm chuẩn bị dữ liệu cho SVR
def prepare_svr_data(df, time_step=60):
    data = df['Close'].values.reshape(-1, 1)

    # Chuẩn hóa dữ liệu với MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x_data, y_data = [], []
    for i in range(time_step, len(scaled_data)):
        x_data.append(scaled_data[i-time_step:i, 0])
        y_data.append(scaled_data[i, 0])

    return np.array(x_data), np.array(y_data), scaler

# Hàm huấn luyện SVR
def train_svr(x_train, y_train):
    svr_model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_model.fit(x_train, y_train)
    return svr_model

# Hàm dự báo với SVR
def predict_with_svr(model, x_test, scaler):
    predicted_scaled = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))
    return predicted_prices

# Hàm dự báo tương lai (365 ngày)
def forecast_future_with_svr(model, last_sequence, scaler, days=365):
    future_prices = []
    for _ in range(days):
        next_price_scaled = model.predict(last_sequence.reshape(1, -1))
        next_price = scaler.inverse_transform(next_price_scaled.reshape(-1, 1))
        future_prices.append(next_price[0, 0])

        # Cập nhật dãy cuối cùng với giá dự báo để tiếp tục dự báo
        last_sequence = np.append(last_sequence[1:], next_price_scaled)

    return future_prices

# Tạo danh sách để lưu kết quả
forecast_results = []

for symbol in symbols:
    print(f"Processing {symbol}...")

    # Load dữ liệu
    df = load_stock_data(symbol)

    # Chuẩn bị dữ liệu
    x_data, y_data, scaler = prepare_svr_data(df)

    # Chia dữ liệu train/test (80% train, 20% test)
    split = int(len(x_data) * 0.8)
    x_train, y_train = x_data[:split], y_data[:split]
    x_test, y_test = x_data[split:], y_data[split:]

    # Huấn luyện mô hình SVR
    svr_model = train_svr(x_train, y_train)

    # Dự báo trên tập test
    predicted_prices = predict_with_svr(svr_model, x_test, scaler)

    # Chuyển giá trị y_test về giá trị gốc
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Đánh giá các chỉ số
    mse = mean_squared_error(y_test_original, predicted_prices)
    rmse = math.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test_original, predicted_prices)
    mae = np.mean(np.abs(y_test_original - predicted_prices))

    # Tính F1-score bằng cách phân loại xu hướng tăng/giảm
    actual_direction = (y_test_original[1:] > y_test_original[:-1]).astype(int).flatten()
    predicted_direction = (predicted_prices[1:] > predicted_prices[:-1]).astype(int).flatten()
    f1 = f1_score(actual_direction, predicted_direction)

    print(f"Completed processing for {symbol}. RMSE: {rmse}, MSE: {mse}, MAPE: {mape}, F1 Score: {f1}")

    # Dự báo 1 năm tới
    last_sequence = x_test[-1]  # Sử dụng dãy cuối cùng của tập test để dự báo
    future_prices = forecast_future_with_svr(svr_model, last_sequence, scaler, days=365)

    # Lưu kết quả vào danh sách
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

# Tạo DataFrame tổng hợp kết quả
summary_svr = pd.DataFrame(forecast_results)

# Chuyển các cột dự đoán và giá trị thực tế thành chuỗi để dễ lưu trữ
summary_svr['Predicted_Prices'] = summary_svr['Predicted_Prices'].apply(lambda x: ', '.join(map(str, x)))
summary_svr['Actual_Prices'] = summary_svr['Actual_Prices'].apply(lambda x: ', '.join(map(str, x)))
summary_svr['Future_Price_Predictions'] = summary_svr['Future_Price_Predictions'].apply(lambda x: ', '.join(map(str, x)))

# Lưu DataFrame vào file CSV
output_file = os.path.join(os.path.dirname(__file__), 'forecast_summary_svr.csv')
summary_svr.to_csv(output_file, index=False)