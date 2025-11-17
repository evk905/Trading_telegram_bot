import json
import numpy as np
from scipy.signal import find_peaks

from config import FORECAST_HORIZON, VAL_SIZE
from models.rf import train_random_forest_on_history
from models.rf_pred import rf_forecast_30_days
from models.arima import  train_arima_on_history_log
from models.lstm import train_lstm_on_history
from models.arima_pred import forecast_30_days_arima_log 
from models.lstm_pred import lstm_predict_future_30_days


def select_best_model(df, ticker):
   
    # Обучение моделей
    metrics_rf, path_rf = train_random_forest_on_history(df, ticker)
    metrics_arima, path_arima = train_arima_on_history_log(df, ticker)
    metrics_lstm, path_lstm = train_lstm_on_history(df, ticker)
    
    # Сравнение по MAE (предполагаем, что метрики - это словари с ключом 'MAE')
    mae_rf = metrics_rf.get('mae', float('inf'))
    mae_arima = metrics_arima.get('mae', float('inf'))
    mae_lstm = metrics_lstm.get('mae', float('inf'))
    
    # Выбор лучшей модели по минимальному MAE
    mae_values = {'RF': mae_rf, 'ARIMA': mae_arima, 'LSTM': mae_lstm}
    best_model = min(mae_values, key=mae_values.get)
    best_path = {'RF': path_rf, 'ARIMA': path_arima, 'LSTM': path_lstm}[best_model]
    
    print(f"Выбрана лучшая модель: {best_model} с MAE: {mae_values[best_model]:.4f}")
    return best_model, best_path
    
    
    
def forecast(df, ticker, amount, best_model, best_path):    
    # Прогноз на 30 дней в зависимости от лучшей модели
    if best_model == 'RF':
        forecast_info = rf_forecast_30_days(df, ticker, model_artifact_path=best_path)
    elif best_model == 'ARIMA':
        forecast_info = forecast_30_days_arima_log(df, ticker, model_artifact_path=best_path)
    elif best_model == 'LSTM':
        forecast_info = lstm_predict_future_30_days(df, ticker, model_artifact_path=best_path)
    
    # Вывод графика и summary
    print(forecast_info["summary"])
    print("График сохранён в:", forecast_info["plot_path"])
    
    # Анализ предсказанного ряда на локальные max/min
    # Предполагаем, что forecast_info содержит 'forecast' - массив предсказанных значений на 30 дней
    forecast_prices = np.array(forecast_info['y_pred'])  # Замените на реальный ключ, если отличается
    
    # Поиск локальных максимумов (пики)
    max_indices, _ = find_peaks(forecast_prices)
    # Поиск локальных минимумов (впадины) - инвертируем ряд
    min_indices, _ = find_peaks(-forecast_prices)
    
    # Рекомендации
    recommendations = []
    for idx in sorted(np.concatenate([max_indices, min_indices])):
        price = forecast_prices[idx]
        day = idx + 1  # День от 1 до 30
        if idx in max_indices:
            rec = f"День {day}: Локальный максимум ({price:.2f}) - Рекомендация: Продавать"
        else:
            rec = f"День {day}: Локальный минимум ({price:.2f}) - Рекомендация: Покупать"
        recommendations.append(rec)
        print(rec)
    
    # Симуляция торговли для расчета условного дохода
    # Простая стратегия: покупаем на минимумах, продаем на максимумах (предполагаем последовательные пары)
    # Сортируем минимумы и максимумы по времени
    trades = []
    min_list = sorted(min_indices)
    max_list = sorted(max_indices)
    
    # Простая симуляция: пары покупка-продажа (игнорируем несбалансированные)
    num_trades = min(len(min_list), len(max_list))
    profit = 0
    shares = amount / forecast_prices[min_list[0]] if min_list else 0  # Начальная покупка, если есть минимум
    
    for i in range(num_trades):
        buy_price = forecast_prices[min_list[i]]
        sell_price = forecast_prices[max_list[i]]
        if sell_price > buy_price:
            trade_profit = (sell_price - buy_price) / buy_price * amount
            profit += trade_profit
            print(f"Торговля {i+1}: Покупка по {buy_price:.2f}, продажа по {sell_price:.2f}. Прибыль: {trade_profit:.2f}")
    
    # Сводка
    summary = {
        "strategy": "Стратегия: Покупать на локальных минимумах, продавать на локальных максимумах в предсказанном периоде.",
        "estimated_profit": f"Ориентировочная прибыль от {amount:.2f} инвестиций: {profit:.2f}"
    }
    print("Сводка:")
    print(summary["strategy"])
    print(summary["estimated_profit"])
    
    # Возврат forecast_info для дальнейшего использования
    return forecast_info, summary

