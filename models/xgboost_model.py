import xgboost as xgb
import numpy as np
import math
import os
import sys
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, f1_score
from xgboost import XGBRegressor

# Thêm đường dẫn thư mục cha để có thể import stock_data_processing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import các biến và hàm từ stock_data_processing.py
from stock_data_processing import folder_path, symbols

# Tạo danh sách chứa kết quả dự báo và các chỉ số
forecast_results = []

# Hàm load dữ liệu
def load_stock_data_for_xgboost(symbol):
    file_path = os.path.join(folder_path, f'{symbol}.csv')
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']  # Đổi tên cột cho nhất quán với yêu cầu của Prophet
    return df

# Hàm chuẩn bị dữ liệu cho XGBoost
def prepare_data_xgboost(df):
    df['ds'] = pd.to_datetime(df['ds'])
    df['days'] = (df['ds'] - df['ds'].min()).dt.days  # Sử dụng số ngày để làm biến dự báo
    X = df[['days']].values
    y = df['y'].values
    return X, y

# Dự báo bằng XGBoost và tính toán các chỉ số
for symbol in symbols:
    print(f"Processing {symbol}...")

    # Load dữ liệu
    df = load_stock_data_for_xgboost(symbol)

    # Chia dữ liệu thành train và test
    train_size = int(len(df) * 0.8)
    train_data = df[:train_size]
    test_data = df[train_size:]

    # Chuẩn bị dữ liệu cho mô hình
    X_train, y_train = prepare_data_xgboost(train_data)
    X_test, y_test = prepare_data_xgboost(test_data)

    # Khởi tạo model XGBoost
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Dự báo trên tập test
    predicted_prices = model.predict(X_test)

    # Tính toán các chỉ số
    mse = mean_squared_error(y_test, predicted_prices)
    rmse = math.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, predicted_prices)

    # Tính F1-score bằng cách phân loại xu hướng tăng/giảm
    actual_direction = (y_test[1:] > y_test[:-1]).astype(int)
    predicted_direction = (predicted_prices[1:] > predicted_prices[:-1]).astype(int)
    f1 = f1_score(actual_direction, predicted_direction)

    # Chuẩn bị dữ liệu cho dự báo tương lai 365 ngày
    last_day = df['ds'].max()
    future_days = np.array([(last_day + pd.Timedelta(days=i)).dayofyear for i in range(1, 366)]).reshape(-1, 1)

    # Dự báo 365 ngày tới
    future_forecast = model.predict(future_days)

    # Lưu kết quả dự báo và các chỉ số vào danh sách
    forecast_results.append({
        'Symbol': symbol,
        'RMSE': rmse,
        'MSE': mse,
        'MAPE': mape,
        'F1 Score': f1,
        'Predicted_Prices': predicted_prices.tolist(),
        'Actual_Prices': y_test.tolist(),
        'Forecast_2024': future_forecast.tolist()
    })

# Tạo DataFrame từ kết quả tổng hợp
summary_xg = pd.DataFrame(forecast_results)

# Chuyển các cột dự đoán và giá trị thực tế thành chuỗi để dễ lưu trữ
summary_xg['Predicted_Prices'] = summary_xg['Predicted_Prices'].apply(lambda x: ', '.join(map(str, x)))
summary_xg['Actual_Prices'] = summary_xg['Actual_Prices'].apply(lambda x: ', '.join(map(str, x)))
summary_xg['Forecast_2024'] = summary_xg['Forecast_2024'].apply(lambda x: ', '.join(map(str, x)))

# Lưu DataFrame vào file CSV
output_file = os.path.join(os.path.dirname(__file__), 'forecast_summary_xgboost.csv')
summary_xg.to_csv(output_file, index=False)