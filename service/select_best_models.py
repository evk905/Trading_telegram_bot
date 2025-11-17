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
    best_mae = mae_values[best_model]

    print(f"Выбрана лучшая модель: {best_model} с MAE: {best_mae :.4f}")
    return best_model, best_path,  best_mae 
    
    
    
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
    
    # Возврат forecast_info и сводку для дальнейшего использования
    return forecast_info, summary



# Функция forecast для бота
def forecast_bot(df, ticker, amount, best_model, best_path):
    # Прогноз на 30 дней
    if best_model == 'RF':
        forecast_info = rf_forecast_30_days(df, ticker, model_artifact_path=best_path)
    elif best_model == 'ARIMA':
        forecast_info = forecast_30_days_arima_log(df, ticker, model_artifact_path=best_path)
    elif best_model == 'LSTM':
        forecast_info = lstm_predict_future_30_days(df, ticker, model_artifact_path=best_path)
    else:
        raise ValueError(f"Неизвестная модель: {best_model}")

    summary_text = forecast_info["summary"]
    plot_path = forecast_info["plot_path"]

    # Ряд прогнозных цен
    forecast_prices = np.array(forecast_info['y_pred'])

    # Локальные максимумы и минимумы
    max_indices, _ = find_peaks(forecast_prices)
    print(max_indices, _)
    min_indices, _ = find_peaks(-forecast_prices)
    print(min_indices, _)

    max_indices = sorted(max_indices.tolist())
    print(max_indices)
    min_indices = sorted(min_indices.tolist())
    print(min_indices)

    # Обработка случая, когда нет локальных максимумов и минимумов (монотонная функция)
    if not max_indices and not min_indices:
      global_min_idx = np.argmin(forecast_prices)
      global_max_idx = np.argmax(forecast_prices)
      min_indices = [global_min_idx]
      max_indices = [global_max_idx]
      # Сортируем заново для последовательности
      min_indices = sorted(min_indices)
      max_indices = sorted(max_indices)
      print("Монотонный ряд: глобальный минимум на индексе", min_indices)
      print("Монотонный ряд: глобальный максимум на индексе", max_indices)

    recommendations = []

    # Для наглядности выведем все локальные экстремумы (не только те, по которым торгуем)
    for idx in sorted(max_indices + min_indices):
        price = forecast_prices[idx]
        day = idx + 1
        if idx in max_indices:
            rec = f"День {day}: Локальный максимум ({price:.2f}) — рекомендация: продавать."
        else:
            rec = f"День {day}: Локальный минимум ({price:.2f}) — рекомендация: покупать."
        recommendations.append(rec)

    # === СИМУЛЯЦИЯ ТОРГОВЛИ ПО СХЕМЕ МИН → МАКС → МИН → МАКС... ===

    trades = []
    profit = 0.0

    if len(min_indices) == 0 or len(max_indices) == 0:
        # Нет либо минимумов, либо максимумов — торговать не по чему
        summary = {
            "strategy": "Стратегия: локальные экстремумы не найдены или их недостаточно для торговли.",
            "estimated_profit_value": 0.0,
            "estimated_profit": (
                f"Ориентировочная прибыль от {amount:.2f} инвестиций: 0.00"
            ),
        }
        return forecast_info, summary, recommendations, trades, plot_path, summary_text

    # Объединяем экстремумы в общий список с типами
    # ('min' / 'max', index)
    extrema = []
    for i in min_indices:
        extrema.append(("min", i))
    for i in max_indices:
        extrema.append(("max", i))
    # сортируем по дню (индексу)
    extrema.sort(key=lambda x: x[1])

    # Стратегия:
    # ищем первый минимум, после него первый максимум, совершаем сделку,
    # далее снова минимум после этого максимума и т.д.
    i = 0
    n = len(extrema)

    # ищем первый минимум
    while i < n and extrema[i][0] != "min":
        i += 1

    while i < n:
        # текущий минимум
        if extrema[i][0] != "min":
            i += 1
            continue
        min_idx = extrema[i][1]
        buy_price = forecast_prices[min_idx]

        # ищем следующий максимум после этого минимума
        j = i + 1
        max_idx = None
        while j < n:
            if extrema[j][0] == "max":
                max_idx = extrema[j][1]
                break
            j += 1

        if max_idx is None:
            # больше максимумов нет — цикл заканчиваем
            break

        sell_price = forecast_prices[max_idx]

        # Совершаем сделку только если рост цены
        if sell_price > buy_price:
            # покупаем на всю сумму amount
            shares = amount / buy_price
            sell_amount = shares * sell_price
            trade_profit = sell_amount - amount
            profit += trade_profit

            trades.append(
                f"Торговля {len(trades) + 1}: "
                f"покупка в день {min_idx + 1} по цене {buy_price:.2f}, "
                f"продажа в день {max_idx + 1} по цене {sell_price:.2f}. "
                f"Прибыль: {trade_profit:.2f}"
            )

        # продолжаем поиск следующей пары минимум–максимум после этого максимума
        i = j + 1

    # Сводка по стратегии
    if trades:
        strategy_text = "Стратегия:\n" + "\n".join(trades)
    else:
        strategy_text = (
            "Стратегия: по найденным локальным максимумам и минимумам "
            "не удалось сформировать прибыльные сделки в пределах 30 дней."
        )

    summary = {
        "strategy": strategy_text,
        "estimated_profit_value": round(float(profit), 2),
        "estimated_profit": (
            f"Ориентировочная прибыль от {amount:.2f} инвестиций: {profit:.2f}"
        ),
    }

    return forecast_info, summary, recommendations, trades, plot_path, summary_text