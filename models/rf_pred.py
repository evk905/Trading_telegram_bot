import os
import sys
import json
import joblib
from typing import Dict, Any, Union
import typing as t
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from config import FORECAST_HORIZON, VAL_SIZE
from metrics_plots import metrics, plot_and_save

import matplotlib.pyplot as plt
from datetime import timedelta

def _recompute_features_for_last_row(hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Пересчёт всех признаков как в make_features, но на всём hist_df без dropna.
    Возвращает последнюю строку с признаками (для прогноза следующего дня).
    Требования:
      - колонки: ['Open','High','Low','Close','AdjClose','Volume'] и опционально 'Date'
    """
    df = hist_df.copy()

    # Лаги по AdjClose
    for lag in [1, 2, 3, 5, 10]:
        df[f"AdjClose_lag{lag}"] = df["AdjClose"].shift(lag)

    # Доходность
    df["ret_1"] = df["AdjClose"].pct_change()

    # Скользящие статистики
    for w in [3, 5, 10, 20]:
        df[f"sma_{w}"] = df["AdjClose"].rolling(w).mean()
        df[f"std_{w}"] = df["AdjClose"].rolling(w).std()
        df[f"ret_mean_{w}"] = df["ret_1"].rolling(w).mean()
        df[f"ret_std_{w}"] = df["ret_1"].rolling(w).std()

    # Диапазон дня и изменения O-C
    df["hl_range"] = (df["High"] - df["Low"]) / df["Open"].replace(0, np.nan)
    df["oc_change"] = (df["Close"] - df["Open"]) / df["Open"].replace(0, np.nan)

    # Объемы и их статистики
    df["vol_chg"] = df["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    for w in [5, 20]:
        df[f"vol_sma_{w}"] = df["Volume"].rolling(w).mean()
        df[f"vol_std_{w}"] = df["Volume"].rolling(w).std()

    # Берём последнюю строку
    last_row = df.iloc[[-1]].copy()
    return last_row

def rf_forecast_30_days(
    df: pd.DataFrame,
    ticker: str,
    model_artifact_path: Union[str, Path],
    model_dir: Union[str, Path] = "result/models",
    model_name: str = "RandomForest",
    horizon: int = FORECAST_HORIZON
) -> Dict[str, Any]:
    """
    Делает 30-дневный рекурсивный прогноз следующего дня по обученной модели.
    Сохраняет график и дополняет results JSON полями 'forecast' и 'summary'.
    Возвращает словарь с ключами: 'dates', 'y_pred', 'summary', 'plot_path'.
    """
    model_dir = Path(model_dir)
    model_artifact_path = Path(model_artifact_path)
    art = joblib.load(model_artifact_path)
    rf = art["model"]
    feature_cols = art["feature_cols"]

    # Подготовим рабочую копию данных (только нужные столбцы)
    required_cols = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Не хватает колонки '{col}' в исходных данных для прогноза.")
    work_df = df[required_cols + (["Date"] if "Date" in df.columns else [])].copy()

    # Определяем стартовую дату и последний факт
    if "Date" in work_df.columns:
        # Убедимся, что тип — datetime
        if not np.issubdtype(work_df["Date"].dtype, np.datetime64):
            work_df["Date"] = pd.to_datetime(work_df["Date"])
        last_date = work_df["Date"].iloc[-1]
    else:
        # Если нет 'Date', создадим искусственный индекс по дням
        last_date = pd.Timestamp.today().normalize()
        work_df["Date"] = pd.date_range(end=last_date, periods=len(work_df), freq="B")

    last_adj = float(work_df["AdjClose"].iloc[-1])

    # Сюда будем складывать прогнозы и даты
    forecast_vals = []
    forecast_dates = []

    # Для рекурсии нам нужны предположения о будущих Open/High/Low/Close/Volume.
    # Простые эвристики:
    # - Open_t+1 ≈ AdjClose_t (последняя известная скорректированная цена)
    # - Close_t+1 = Open_t+1
    # - High/Low вокруг Close с небольшим спредом (например, ±0.5%)
    # - Volume — скользящая средняя за 5 дней
    def synthesize_ohlcv(next_base_price: float, hist_slice: pd.DataFrame) -> t.Dict[str, float]:
        vol_ma5 = float(hist_slice["Volume"].tail(5).mean()) if len(hist_slice) >= 5 else float(hist_slice["Volume"].mean())
        # 0.5% диапазон
        high = next_base_price * 1.005
        low = next_base_price * 0.995
        return {
            "Open": next_base_price,
            "High": high,
            "Low": low,
            "Close": next_base_price,
            "AdjClose": next_base_price,  # используем базу как proxy
            "Volume": vol_ma5 if not np.isnan(vol_ma5) else float(hist_slice["Volume"].iloc[-1])
        }

    rolling_df = work_df.copy()

    for i in range(horizon):
        # Пересчитываем признаки на текущем rolling_df и берём последнюю строку фичей
        last_feats = _recompute_features_for_last_row(rolling_df)

        # Проверим, что для последней строки нет NaN по ключевым признакам (первые дни могут быть неполные)
        if last_feats[feature_cols].isna().any(axis=None):
            # Если признаки ещё не стабилизировались (из-за недостатка истории для окон),
            # просто продвинемся добавлением ещё одной "копии" последней факт. строки,
            # чтобы окна заполнились, и продолжим.
            synth_next = synthesize_ohlcv(float(rolling_df["AdjClose"].iloc[-1]), rolling_df)
            next_date = rolling_df["Date"].iloc[-1] + pd.tseries.offsets.BDay()
            synth_next["Date"] = next_date
            rolling_df = pd.concat([rolling_df, pd.DataFrame([synth_next])], ignore_index=True)
            # После добавления снова пересчитаем на следующей итерации
            # и не записываем прогноз в этот раз.
            continue

        X_last = last_feats[feature_cols].astype(float).values
        y_next = float(rf.predict(X_last)[0])

        # Сохраняем прогноз и дату
        next_date = rolling_df["Date"].iloc[-1] + pd.tseries.offsets.BDay()
        forecast_vals.append(y_next)
        forecast_dates.append(pd.Timestamp(next_date))

        # Добавляем синтетическую OHLCV-строку на основе прогноза,
        # чтобы можно было построить признаки для следующего шага.
        synth_next = synthesize_ohlcv(y_next, rolling_df)
        synth_next["Date"] = next_date
        rolling_df = pd.concat([rolling_df, pd.DataFrame([synth_next])], ignore_index=True)

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
    }