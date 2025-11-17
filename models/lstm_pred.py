import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union
import joblib
import json

from config import FORECAST_HORIZON, VAL_SIZE
from metrics_plots import metrics, plot_and_save

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model


def lstm_predict_future_30_days(
    df: pd.DataFrame,
    ticker: str,
    model_artifact_path: Union[str, Path],
    model_dir: Union[str, Path] = "result/models",
    model_name: str = "LSTM",
    horizon: int = FORECAST_HORIZON
) -> Dict[str, Any]:
    model_dir = Path(model_dir)
    model_artifact_path = Path(model_artifact_path)
    art = joblib.load(model_artifact_path)
    keras_path = art['keras_path']
    model = load_model(keras_path)  # Загрузка модели (исправлено)
    feature_cols = art["feature_cols"]
    
    # Подготовка данных
    work_df = df.copy()
    work_df['Date'] = pd.to_datetime(work_df['Date'])
    work_df = work_df.sort_values('Date')
    last_date = work_df['Date'].iloc[-1]
    last_adj = work_df['AdjClose'].iloc[-1]
    
    # Предполагаем, что модель LSTM обучена на унивариантных данных (AdjClose), с scaler и lookback
    scaler = art["scaler"]
    lookback = art["lookback"]
    
    # Получаем исторические данные и масштабируем
    historical_data = work_df['AdjClose'].values
    scaled_historical = scaler.transform(historical_data.reshape(-1, 1))
    
    # Последняя последовательность для прогноза
    if len(scaled_historical) < lookback:
        raise ValueError(f"Недостаточно данных для lookback={lookback}")
    
    last_sequence = scaled_historical[-lookback:].reshape(1, lookback, 1)
    
    # Итеративный прогноз на horizon дней
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    forecast_vals = []
    current_sequence = last_sequence.copy()
    
    for _ in range(horizon):
        pred_scaled = model.predict(current_sequence)[0][0]
        pred = scaler.inverse_transform([[pred_scaled]])[0][0]
        forecast_vals.append(pred)
        
        # Обновляем последовательность: добавляем новое предсказание
        new_scaled = np.array([[pred_scaled]])
        current_sequence = np.append(current_sequence[:, 1:, :], new_scaled.reshape(1, 1, 1), axis=1)
    
    # Построение графика
    plot_path = plot_and_save(
        ticker=ticker,
        dates=work_df["Date"],
        y_hist=work_df["AdjClose"],
        pred_dates=forecast_dates,
        y_pred=forecast_vals,
        model_name=model_name,
        out_dir=str(model_dir),          # сохраняем в ту же папку с моделями/результатами
        forecast_horizon=horizon,        # чтобы в имени файла был ваш горизонт
        dpi=150,
        return_abs=True                  # чтобы в JSON записать абсолютный путь
    )

    # Сообщение о росте/падении относительно текущего дня (берём последний прогноз)
    if len(forecast_vals) > 0:
        last_forecast = forecast_vals[-1]
        abs_change = last_forecast - last_adj
        pct_change = (abs_change / last_adj) * 100 if last_adj != 0 else np.nan
        direction = "вырастут" if abs_change > 0 else ("упадут" if abs_change < 0 else "не изменятся")
        summary = (
            f"Оценка на {forecast_dates[-1].date()}: акции {direction} на "
            f"{abs(abs_change):.2f} ({abs(pct_change):.2f}%) относительно текущего дня."
        )
    else:
        summary = "Недостаточно данных для устойчивого прогноза на 30 дней вперёд."

    # Запись в JSON результатов
    results_json_path = Path(model_dir) / f"{ticker}_{model_name}_results.json"
    results_payload = {}
    if results_json_path.exists():
        with open(results_json_path, "r", encoding="utf-8") as f:
            try:
                results_payload = json.load(f)
            except json.JSONDecodeError:
                results_payload = {}
    results_payload.setdefault("forecast", {})
    results_payload["forecast"]["horizon_days"] = int(horizon)
    results_payload["forecast"]["dates"] = [d.strftime("%Y-%m-%d") for d in forecast_dates]
    results_payload["forecast"]["y_pred"] = [float(v) for v in forecast_vals]
    results_payload["forecast"]["plot_path"] = str(plot_path)
    results_payload["summary"] = summary

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)

    return {
        "dates": forecast_dates,
        "y_pred": forecast_vals,
        "summary": summary,
        "plot_path": str(plot_path),
        'results_json_path': str(results_json_path)
    }