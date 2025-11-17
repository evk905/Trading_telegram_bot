import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# statsmodels ARIMA
from pmdarima.arima import auto_arima
# или можно из statsmodels:
# from statsmodels.tsa.arima.model import ARIMA

from config import  VAL_SIZE
from service.metrics_plots import metrics



def train_arima_on_history_log(
    df: pd.DataFrame,
    ticker: str,
    model_dir: str | Path = "result/models",
    model_name: str = "ARIMA",
    seasonal: bool = True,
    m: int = 5,  # недельная псевдо-сезонность для б/дней
    allow_drift: bool = True,
    max_p: int = 7, max_q: int = 7, max_P: int = 3, max_Q: int = 3, # Увеличенные максимальные порядки
    start_p: int = 1, start_q: int = 1, start_d: int = 1, # Задаем начальные порядки
    start_P: int = 1, start_Q: int = 1, start_D: int = 0, # Задаем начальные сезонные порядки
    val_size: int = VAL_SIZE,
    exog_df: pd.DataFrame = None # Новый параметр для внешних регрессоров
):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Источник ряда
    series = df["AdjClose"].astype(float)
    dates = pd.to_datetime(df["Date"]) if "Date" in df.columns else pd.bdate_range(periods=len(df), end=pd.Timestamp.today().normalize())

    # Разделение на train/valid (валидация только на хвосте)
    n = len(series)
    val_size = int(val_size) if val_size is not None else 0
    valid_start = max(0, n - val_size)
    train_series = series.iloc[:valid_start].copy()
    y_valid = series.iloc[valid_start:].dropna().values
    valid_idx = np.arange(valid_start, valid_start + len(y_valid))

    # Лог-преобразование
    eps = 1e-9
    y_log_train = np.log(train_series.values + eps)

    # Подготовка exog-данных
    exog_train = exog_df.iloc[:valid_start] if exog_df is not None else None
    exog_valid = exog_df.iloc[valid_start:] if exog_df is not None else None

    # Автоподбор ARIMA на лог-уровне треня
    arima_model = auto_arima(
        y_log_train,
        X=exog_train,
        seasonal=seasonal,
        m=m,
        information_criterion="aic",
        stepwise=True, # <-- ИЗМЕНИТЬ ЗДЕСЬ
        suppress_warnings=True,
        error_action="ignore",
        max_p=max_p, max_q=max_q, max_d=1,
        max_P=max_P, max_Q=max_Q, max_D=1 if seasonal else 0,
        start_p=start_p, start_q=start_q, start_d=start_d,
        start_P=start_P, start_Q=start_Q, start_D=start_D,
        max_order=None,
        trace=False,
        with_intercept=True,
        trend="c",
        allowdrift=allow_drift,
        n_jobs=-1 # <-- ДОБАВИТЬ ЗДЕСЬ
    )


    # Walk-forward на валидации: предсказываем лог-уровень на 1 шаг вперёд, обновляем модель фактом (на лог-уровне)
    y_pred_valid = []
    for t in range(len(y_valid)):
        # прогноз лог-цены на 1 шаг
        # Учитываем exog для прогноза, если он есть
        fc_log = arima_model.predict(n_periods=1, X=exog_valid.iloc[t:t+1] if exog_valid is not None else None)[0]
        # обратно в цены
        fc_price = float(np.exp(fc_log) - eps)
        y_pred_valid.append(fc_price)

        # Обновляем модель фактическим лог-значением текущего дня валидации
        current_real_adjclose = float(series.iloc[valid_idx[t]])
        # Учитываем exog для обновления, если он есть
        arima_model.update([np.log(current_real_adjclose + eps)], X=exog_valid.iloc[t:t+1] if exog_valid is not None else None)

    y_pred_valid = np.array(y_pred_valid, dtype=float)


    # Метрики
    rmse, mae, mape, r2 = metrics(ticker, y_true=y_valid, y_pred=np.array(y_pred_valid), model_name=model_name, model_dir=model_dir)

    # Обучим финальную модель на всём ряде (на лог-уровне) для продакшн-прогноза
    y_log_full = np.log(series.values + eps)
   
    final_model = auto_arima(
        y_log_full,
        X=exog_df,
        seasonal=seasonal,
        m=m,
        information_criterion="aic",
        stepwise=True, # <-- ИЗМЕНИТЬ ЗДЕСЬ
        suppress_warnings=True,
        error_action="ignore",
        max_p=max_p, max_q=max_q, max_d=1,
        max_P=max_P, max_Q=max_Q, max_D=1 if seasonal else 0,
        start_p=start_p, start_q=start_q, start_d=start_d,
        start_P=start_P, start_Q=start_Q, start_D=start_D,
        max_order=None,
        trace=False,
        with_intercept=True,
        trend="c",
        allowdrift=allow_drift,
        n_jobs=-1 # <-- ДОБАВИТЬ ЗДЕСЬ
    )


    # Сохранение артефакта
    artifact = {
        "model": final_model,
        "transform": {"type": "log", "epsilon": eps},
        "y_last": float(series.values[-1]),
        "y_log_last": float(y_log_full[-1]),
        "last_date": str(pd.to_datetime(dates).iloc[-1].date()) if len(dates) else None,
        "ticker": ticker,
        "model_name": model_name,
        "val_size": val_size,
        "exog_cols": list(exog_df.columns) if exog_df is not None else None # Сохраняем имена колонок exog
    }

    model_path = model_dir / f"{ticker}_{model_name}.joblib"
    joblib.dump(artifact, model_path)

    # results.json: добавим как метрики AIC/BIC (для информации), так и вал-метрики для сравнения
    results_json_path = model_dir / f"{ticker}_{model_name}_results.json"
    payload = {}
    if results_json_path.exists():
        try:
            payload = json.loads(Path(results_json_path).read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    payload["metrics"] = {
        "rmse": rmse, "mae": mae, "mape": mape, "r2": r2
    }
    # Информативные параметры модели (с треня на всем ряде)
    payload["model_info"] = {
        "order": tuple(int(x) for x in final_model.order),
        "seasonal_order": tuple(int(x) for x in final_model.seasonal_order) if seasonal else (0,0,0,0),
        "aic": float(final_model.aic()),
        "bic": float(final_model.bic()),
        "trend": "c", # Изменено на "c"
        "allow_drift": allow_drift,
        "m": m
    }
    payload["val_size"] = int(val_size)
    payload["feature_count"] = 1 + (len(exog_df.columns) if exog_df is not None else 0)
    payload["cv"] = None

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Возвращаем вал-метрики, как требуется для сравнения с другими моделями
    return payload["metrics"] , str(model_path)