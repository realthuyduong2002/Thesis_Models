import os
import sys
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, f1_score
import math

# Thêm đường dẫn thư mục cha để có thể import stock_data_processing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import các biến và hàm từ stock_data_processing.py
from stock_data_processing import folder_path, symbols

# Tải dữ liệu và chuẩn bị dữ liệu cho NeuralProphet
def prepare_neuralprophet_data(df):
    # Đổi tên cột cho đúng định dạng của NeuralProphet
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    return df

# Huấn luyện mô hình NeuralProphet
def build_neuralprophet_model():
    model = NeuralProphet(daily_seasonality=True)  # Kích hoạt tính mùa vụ hàng ngày
    return model

# Đọc dữ liệu từ CSV cho một mã cổ phiếu
def load_stock_data(symbol):
    file_path = os.path.join(folder_path, f'{symbol}.csv')
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df[['Date', 'Close']]
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})  # Đổi tên cột cho phù hợp với NeuralProphet
    return df

# Tạo danh sách để lưu kết quả
forecast_results = []

# Dự báo bằng NeuralProphet
for symbol in symbols:
    print(f"Processing {symbol}...")
    
    # Load dữ liệu
    df = load_stock_data(symbol)
    
    # Tạo mô hình NeuralProphet
    model = build_neuralprophet_model()
    
    # Huấn luyện mô hình
    model.fit(df, freq='D')
    
    # Dự báo 365 ngày tiếp theo
    future = model.make_future_dataframe(df, periods=365)
    forecast = model.predict(future)
    
    # Tính toán các chỉ số trên dữ liệu gốc
    y_test = df['y'].values[-365:]  # Lấy 365 giá trị cuối cùng để tính toán
    y_pred = forecast['yhat1'].values[-365:]  # Giá trị dự đoán

    # Tính toán các chỉ số
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))

    # Tính F1-score bằng cách phân loại tăng/giảm
    actual_direction = (y_test[1:] > y_test[:-1]).astype(int)
    predicted_direction = (y_pred[1:] > y_pred[:-1]).astype(int)
    f1 = f1_score(actual_direction, predicted_direction)

    # Lưu kết quả vào danh sách
    forecast_results.append({
        'Symbol': symbol,
        'RMSE': rmse,
        'MSE': mse,
        'MAPE': mape,
        'MAE': mae,
        'F1 Score': f1,
        'Predicted_Prices': y_pred.tolist(),  # Giá trị dự đoán
        'Actual_Prices': y_test.tolist(),      # Giá trị thực tế
        'Forecast_2024': forecast['yhat1'].values[-365:].tolist()  # Dự đoán 365 ngày tiếp theo
    })

# Tạo bảng tổng hợp kết quả dự báo
summary_neuralprophet = pd.DataFrame(forecast_results)

# Chuyển các cột dự đoán và giá trị thực tế thành chuỗi để dễ lưu trữ
summary_neuralprophet['Predicted_Prices'] = summary_neuralprophet['Predicted_Prices'].apply(lambda x: ', '.join(map(str, x)))
summary_neuralprophet['Actual_Prices'] = summary_neuralprophet['Actual_Prices'].apply(lambda x: ', '.join(map(str, x)))
summary_neuralprophet['Forecast_2024'] = summary_neuralprophet['Forecast_2024'].apply(lambda x: ', '.join(map(str, x)))

# Lưu DataFrame vào file CSV
output_file = os.path.join(os.path.dirname(__file__), 'forecast_summary_neuralprophet.csv')
summary_neuralprophet.to_csv(output_file, index=False)