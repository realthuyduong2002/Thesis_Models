import os
import sys
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, f1_score
import math

# Thêm đường dẫn thư mục cha để có thể import stock_data_processing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import các biến và hàm từ stock_data_processing.py
from stock_data_processing import folder_path, symbols

# Hàm đọc dữ liệu và chuẩn bị cho Prophet
def load_stock_data_for_prophet(symbol):
    file_path = os.path.join(folder_path, f'{symbol}.csv')
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']  # Prophet yêu cầu cột 'ds' cho ngày và 'y' cho giá trị
    return df

# Tạo danh sách chứa kết quả dự báo và các chỉ số
forecast_results = []

# Dự báo bằng Prophet và tính toán các chỉ số
for symbol in symbols:
    print(f"Processing {symbol}...")

    # Load dữ liệu
    df = load_stock_data_for_prophet(symbol)

    # Chia dữ liệu thành train và test
    train_size = int(len(df) * 0.8)
    train_data = df[:train_size]
    test_data = df[train_size:]

    # Khởi tạo model Prophet
    model = Prophet()
    model.fit(train_data)

    # Dự báo cho 365 ngày trong tương lai
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # Chỉ giữ lại các ngày trong tập test để đánh giá
    test_pred = forecast[['ds', 'yhat']].iloc[train_size:train_size + len(test_data)]
    test_actual = test_data[['ds', 'y']]

    # Tính toán các chỉ số
    mse = mean_squared_error(test_actual['y'], test_pred['yhat'])
    rmse = math.sqrt(mse)
    mape = mean_absolute_percentage_error(test_actual['y'], test_pred['yhat'])
    mae = np.mean(np.abs(test_actual['y'] - test_pred['yhat']))

    # Tính F1-score bằng cách phân loại xu hướng tăng/giảm
    actual_direction = (test_actual['y'].values[1:] > test_actual['y'].values[:-1]).astype(int)
    predicted_direction = (test_pred['yhat'].values[1:] > test_pred['yhat'].values[:-1]).astype(int)
    f1 = f1_score(actual_direction, predicted_direction)

    # Tạo kết quả cho dự báo tương lai 365 ngày sau 1/1/2024
    forecast_2024 = forecast[forecast['ds'] > pd.to_datetime('2024-01-01')][['ds', 'yhat']]

    # Lưu kết quả dự báo và các chỉ số vào danh sách
    forecast_results.append({
        'Symbol': symbol,
        'RMSE': rmse,
        'MSE': mse,
        'MAPE': mape,
        'MAE': mae,
        'F1 Score': f1,
        'Predicted_Prices': test_pred['yhat'].values.tolist(),
        'Actual_Prices': test_actual['y'].values.tolist(),
        'Forecast_2024': forecast_2024['yhat'].values.tolist()
    })

# Tạo DataFrame tổng hợp kết quả
summary_prophet = pd.DataFrame(forecast_results)

# Chuyển các cột dự đoán và giá trị thực tế thành chuỗi để dễ lưu trữ
summary_prophet['Predicted_Prices'] = summary_prophet['Predicted_Prices'].apply(lambda x: ', '.join(map(str, x)))
summary_prophet['Actual_Prices'] = summary_prophet['Actual_Prices'].apply(lambda x: ', '.join(map(str, x)))
summary_prophet['Forecast_2024'] = summary_prophet['Forecast_2024'].apply(lambda x: ', '.join(map(str, x)))

# Lưu DataFrame vào file CSV duy nhất
output_file = os.path.join(os.path.dirname(__file__), 'forecast_summary_prophet.csv')
summary_prophet.to_csv(output_file, index=False)