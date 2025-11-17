import os
from datetime import datetime
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima

from service.metrics_plots import metrics, plot_and_save
from config import VAL_SIZE, FORECAST_HORIZON



def forecast_30_days_arima_log(
    df: pd.DataFrame,
    ticker: str,
    model_artifact_path: str | Path,
    model_dir: str | Path = "result/models",
    model_name: str = "ARIMA",
    horizon: int = FORECAST_HORIZON,
    use_business_days: bool = True,
    connect_last_point: bool = True, # Этот параметр не используется в функции
    exog_forecast_df: pd.DataFrame = None # Новый параметр для внешних регрессоров прогноза
):
    from copy import deepcopy
    model_dir = Path(model_dir)
    model_artifact_path = Path(model_artifact_path)
    art = joblib.load(model_artifact_path)
    base_model = art["model"]
    eps = art["transform"]["epsilon"]
    exog_cols = art.get("exog_cols") # Получаем имена колонок exog из артефакта

    # История
    dates_hist = pd.to_datetime(df["Date"]) if "Date" in df.columns else pd.bdate_range(periods=len(df), end=pd.Timestamp.today().normalize())
    y_hist = df["AdjClose"].astype(float).values
    # y_log_hist = np.log(y_hist + eps) # Не используется напрямую в прогнозе, только для инициализации

    # Сгенерируем даты прогноза
    start_next = dates_hist.iloc[-1] + (pd.tseries.offsets.BDay() if use_business_days else pd.Timedelta(days=1))
    pred_dates = (pd.bdate_range(start=start_next, periods=horizon) if use_business_days
                  else pd.date_range(start=start_next, periods=horizon, freq="D"))

    # Проверка и подготовка exog для прогноза
    exog_for_forecast = None
    if exog_cols is not None:
        if exog_forecast_df is None:
            raise ValueError("Модель была обучена с exog-переменными. Необходимо предоставить 'exog_forecast_df' для прогнозирования.")
        
        # Убедимся, что exog_forecast_df содержит нужные колонки
        missing_cols = [col for col in exog_cols if col not in exog_forecast_df.columns]
        if missing_cols:
            raise ValueError(f"exog_forecast_df не содержит необходимые колонки из обучения: {missing_cols}")
        
        # Выбираем только нужные колонки и убеждаемся, что они расположены в правильном порядке
        exog_for_forecast = exog_forecast_df[exog_cols].copy()
        
        # Убедимся, что exog_for_forecast имеет правильную длину для горизонта прогноза
        if len(exog_for_forecast) < horizon:
            raise ValueError(f"exog_forecast_df содержит {len(exog_for_forecast)} записей, но горизонт прогноза {horizon}. Недостаточно данных.")
        exog_for_forecast = exog_for_forecast.iloc[:horizon]

    model = deepcopy(base_model)  # чтобы не портить артефакт
    y_log_preds = []
    # cur_level = float(y_log_hist[-1]) # Эта переменная не нужна при использовании model.update()

    # Пошаговый прогноз: предсказываем дельту уровня и обновляем
    for i in range(horizon):
        # 1 шаг вперёд
        # Передаем соответствующий exog для каждого шага прогноза
        exog_step = exog_for_forecast.iloc[i:i+1] if exog_for_forecast is not None else None
        step_pred_log = model.predict(n_periods=1, X=exog_step)[0]
        y_log_preds.append(step_pred_log)
        
        # Обновляем модель «наблюдением» шага — принимаем предсказанное значение как наблюдение
        model.update([step_pred_log], X=exog_step)

    forecast_vals  = np.exp(np.array(y_log_preds)) - eps

    # График
    plot_path = plot_and_save(
        ticker=ticker,
        dates=dates_hist,
        y_hist=y_hist,
        pred_dates=pred_dates,
        y_pred= forecast_vals ,
        model_name=model_name,
        out_dir=str(model_dir),
        forecast_horizon=horizon,
        dpi=150,
        return_abs=True,
    )
  
    # Summary
    last_adj = float(y_hist[-1])
    last_forecast = float(forecast_vals [-1])
    abs_change = last_forecast - last_adj
    pct_change = (abs_change / last_adj) * 100 if last_adj != 0 else np.nan
    direction = "вырастут" if abs_change > 0 else ("упадут" if abs_change < 0 else "не изменятся")
    summary = (
        f"Оценка на {pred_dates[-1].date()}: акции {direction} на "
        f"{abs(abs_change):.2f} ({abs(pct_change):.2f}%) относительно текущего дня."
    )
   
    # Обновление JSON результатов
    results_json_path = model_dir / f"{ticker}_{model_name}_results.json"
    results_payload = {}
    if results_json_path.exists():
        with open(results_json_path, "r", encoding="utf-8") as f:
            try:
                results_payload = json.load(f)
            except json.JSONDecodeError:
                results_payload = {}

    results_payload.setdefault("forecast", {})
    results_payload["forecast"]["horizon_days"] = int(horizon)
    results_payload["forecast"]["dates"] = [d.strftime("%Y-%m-%d") for d in pred_dates]
    results_payload["forecast"]["y_pred"] = [float(v) for v in  forecast_vals ]
    results_payload["forecast"]["plot_path"] = str(plot_path)
    results_payload["summary"] = summary

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)

    return {
        "dates": pred_dates,
        "y_pred":  forecast_vals ,
        "summary": summary,
        "plot_path": str(plot_path),
    }