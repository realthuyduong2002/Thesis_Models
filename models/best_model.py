import pandas as pd

# Đường dẫn đến các file CSV của từng mô hình
model_files = {
    'LSTM': 'forecast_summary_lstm.csv',
    'SVR': 'forecast_summary_svr.csv',
    'Prophet': 'forecast_summary_prophet.csv',
    'XGBoost': 'forecast_summary_xgboost.csv',
    'NeuralProphet': 'forecast_summary_neuralprophet.csv',
    'Hybrid_LSTM_SVR': 'forecast_summary_hybrid.csv'
}

# Tạo DataFrame trống để lưu kết quả tổng hợp
final_results = pd.DataFrame()

# Đọc từng file CSV và thêm thông tin về mô hình
model_dfs = {}
for model_name, file_path in model_files.items():
    model_df = pd.read_csv(file_path)
    model_df['Model'] = model_name  # Thêm cột để nhận biết mô hình
    model_dfs[model_name] = model_df

# Kết hợp các DataFrame thành một DataFrame duy nhất
all_results = pd.concat(model_dfs.values(), ignore_index=True)

# Hàm để chọn mô hình tốt nhất dựa trên các chỉ số
def select_best_model(group):
    # Tạo cột điểm khởi đầu cho mỗi mô hình
    group['Score'] = 0

    # Kiểm tra điều kiện tốt nhất cho từng chỉ số và tăng điểm cho mô hình thỏa mãn điều kiện
    # Giảm thiểu: RMSE, MSE, MAPE, MAE (thấp hơn là tốt hơn)
    group.loc[group['RMSE'] == group['RMSE'].min(), 'Score'] += 1
    group.loc[group['MSE'] == group['MSE'].min(), 'Score'] += 1
    group.loc[group['MAPE'] == group['MAPE'].min(), 'Score'] += 1
    group.loc[group['MAE'] == group['MAE'].min(), 'Score'] += 1

    # Tối đa hóa: F1 Score (cao hơn là tốt hơn)
    group.loc[group['F1 Score'] == group['F1 Score'].max(), 'Score'] += 1

    # Chọn mô hình có điểm số cao nhất (tốt nhất)
    best_model = group.loc[group['Score'].idxmax()]

    return best_model

# Áp dụng hàm chọn mô hình tốt nhất cho mỗi mã chứng khoán
best_models = all_results.groupby('Symbol').apply(select_best_model).reset_index(drop=True)

# Xóa cột `Forecast_2024` trước khi lưu vào file CSV
best_models = best_models.drop(columns=['Forecast_2024'])

# Lưu kết quả vào file CSV
best_models.to_csv('best_model_for_each_symbol.csv', index=False)

# In ra kết quả
print(best_models)
