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
from metrics_plots import metrics



def train_arima_on_history(
    df: pd.DataFrame,
    ticker: str,
    model_dir: str | Path = "result/models",
    model_name: str = "ARIMA",
    val_size: int = VAL_SIZE,
    arima_kwargs: dict | None = None
):
    """
    Обучает ARIMA на ряде AdjClose с целью AdjClose[t+1], считает метрики на валидации
    и сохраняет модель в файл. Возвращает: (dict метрик, путь к модели).
    Примечание: для честного сравнения используем тот же временной split (по индексу),
    что и у RandomForest, но ARIMA обучается на целевой серии и валидируется одношагово.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    series = df["AdjClose"].astype(float).reset_index(drop=True)

    # Создадим цель y = AdjClose.shift(-1) в терминах индексов
    y = series.shift(-1)
    valid_start = len(series) - val_size
    if valid_start <= 30:
        raise ValueError("Слишком мало данных для выбранного val_size")

    # Обучение только на train отрезке
    train_series = series.iloc[:valid_start].copy()
    y_valid = y.iloc[valid_start:-0 if val_size == 0 else None].dropna().values
    
    # Для совместимости индексы валидации:
    valid_idx = np.arange(valid_start, valid_start + len(y_valid))

    # Подбор/обучение ARIMA
    # Можно использовать pmdarima.auto_arima для быстрой и устойчивой настройки
    default_kwargs = dict(
        start_p=0, start_q=0, max_p=3, max_q=3,
        seasonal=False,  # дневные котировки без явной сезонности
        d=None,          # авто-оценка дифференцирования
        trace=False, error_action="ignore", suppress_warnings=True,
        stepwise=True, information_criterion="aic"
    )
    if arima_kwargs:
        default_kwargs.update(arima_kwargs)

    arima_model = auto_arima(train_series.values, **default_kwargs)
    
    #  Итеративный форкаст на валидации (walk-forward)
    # Важно: чтобы не допустить утечку, расширяем модель по мере продвижения
    y_pred = []
    history = list(train_series.values)
    for t in range(len(y_valid)):
        # прогноз одного шага вперед
        fc = arima_model.predict(n_periods=1)[0]
        y_pred.append(fc)
        # после «настоящий» факт (AdjClose текущего дня) добавляем в историю
        # внимание: цель — AdjClose[t+1], поэтому в history добавляем текущий AdjClose валидации
        current_real_adjclose = series.iloc[valid_idx[t]]
        arima_model.update(current_real_adjclose)
        history.append(current_real_adjclose)

    # Метрики
    rmse, mae, mape, r2 = metrics(ticker, y_true=y_valid, y_pred=np.array(y_pred), model_name=model_name, model_dir=model_dir)

    # Обучим финальную модель на всем ряде (для production-прогноза следующего дня)
    final_model = auto_arima(series.values, **default_kwargs)

    # Сохранение
    model_path = Path(model_dir) / f"{ticker}_{model_name}.joblib"
    joblib.dump(
        {
            "model": final_model,
            "model_name": model_name,
            "ticker": ticker,
            "val_size": val_size,
            "train_rows": int(valid_start),
            "valid_rows": int(len(y_valid)),
            "last_date": str(df["Date"].max().date()) if "Date" in df.columns else None,
            "note": "pmdarima auto_arima; прогноз AdjClose[t+1]"
        },
        model_path
    )

    results_json = {
        "ticker": ticker,
        "model_name": model_name,
        "metrics": {"rmse": float(rmse), "mae": float(mae), "mape": float(mape), "r2": float(r2)},
        "val_size": int(val_size),
        "feature_count": 1,  # только ряд
        "cv": None
    }
    with open(Path(model_dir) / f"{ticker}_{model_name}_results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    return results_json["metrics"], str(model_path)


# def train_arima_on_history_log_1(
#     df: pd.DataFrame,
#     ticker: str,
#     model_dir: str | Path = "result/models",
#     model_name: str = "ARIMA",
#     seasonal: bool = True,
#     m: int = 5,  # недельная псевдо-сезонность для б/дней
#     allow_drift: bool = True,
#     max_p: int = 5, max_q: int = 5, max_P: int = 2, max_Q: int = 2,
# ):
#     model_dir = Path(model_dir)
#     model_dir.mkdir(parents=True, exist_ok=True)

#     # Источник ряда
#     y = df["AdjClose"].astype(float).values
#     dates = pd.to_datetime(df["Date"]) if "Date" in df.columns else pd.bdate_range(periods=len(df), end=pd.Timestamp.today().normalize())

#     # лог-уровень
#     y_log = np.log(y + 1e-9)

#     # авто-подбор. Важно: включаем trend и drift
#     arima = auto_arima(
#         y_log,
#         seasonal=seasonal,
#         m=m,
#         information_criterion="aic",
#         stepwise=True,
#         suppress_warnings=True,
#         error_action="ignore",
#         max_p=max_p, max_q=max_q,
#         max_P=max_P, max_Q=max_Q,
#         max_order=None,
#         trace=False,
#         with_intercept=True,
#         trend="t",            # линейный тренд
#         allowdrift=allow_drift
#     )

#     artifact = {
#         "model": arima,
#         "transform": {"type": "log", "epsilon": 1e-9},
#         "y_last": float(y[-1]),
#         "y_log_last": float(y_log[-1]),
#         "last_date": str(dates.iloc[-1].date()),
#         "ticker": ticker,
#         "model_name": model_name,
#     }

#     model_path = model_dir / f"{ticker}_{model_name}_arima.joblib"
#     joblib.dump(artifact, model_path)

#     # Простейшие метрики fit-а на трейне (AIC/BIC)
#     metrics = {
#         "aic": float(arima.aic()),
#         "bic": float(arima.bic()),
#         "order": tuple(int(x) for x in arima.order),
#         "seasonal_order": tuple(int(x) for x in arima.seasonal_order) if seasonal else (0,0,0,0),
#         "trend": "t",
#         "allow_drift": allow_drift,
#         "m": m
#     }

#     # results.json дополним/создадим
#     results_json_path = model_dir / f"{ticker}_{model_name}_results.json"
#     payload = {}
#     if results_json_path.exists():
#         try:
#             payload = json.loads(Path(results_json_path).read_text(encoding="utf-8"))
#         except Exception:
#             payload = {}
#     payload["metrics"] = metrics
#     Path(results_json_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

#     return metrics, str(model_path)



# def train_arima_on_history_log_2(
#     df: pd.DataFrame,
#     ticker: str,
#     model_dir: str | Path = "result/models",
#     model_name: str = "ARIMA",
#     seasonal: bool = True,
#     m: int = 5,  # недельная псевдо-сезонность для б/дней
#     allow_drift: bool = True,
#     max_p: int = 5, max_q: int = 5, max_P: int = 2, max_Q: int = 2,
#     val_size: int = VAL_SIZE,
# ):
#     model_dir = Path(model_dir)
#     model_dir.mkdir(parents=True, exist_ok=True)

#     # Источник ряда
#     series = df["AdjClose"].astype(float)
#     dates = pd.to_datetime(df["Date"]) if "Date" in df.columns else pd.bdate_range(periods=len(df), end=pd.Timestamp.today().normalize())

#     # Разделение на train/valid (валидация только на хвосте)
#     n = len(series)
#     val_size = int(val_size) if val_size is not None else 0
#     valid_start = max(0, n - val_size)
#     train_series = series.iloc[:valid_start].copy()
#     y_valid = series.iloc[valid_start:].dropna().values
#     valid_idx = np.arange(valid_start, valid_start + len(y_valid))

#     # Лог-преобразование
#     eps = 1e-9
#     y_log_train = np.log(train_series.values + eps)

#     # Автоподбор ARIMA на лог-уровне треня
#     arima_model = auto_arima(
#         y_log_train,
#         seasonal=seasonal,
#         m=m,
#         information_criterion="aic",
#         stepwise=True,
#         suppress_warnings=True,
#         error_action="ignore",
#         max_p=max_p, max_q=max_q,
#         max_P=max_P, max_Q=max_Q,
#         max_order=None,
#         trace=False,
#         with_intercept=True,
#         trend="t",
#         allowdrift=allow_drift
#     )

#     # Walk-forward на валидации: предсказываем лог-уровень на 1 шаг вперёд, обновляем модель фактом (на лог-уровне)
#     y_pred_valid = []
#     for t in range(len(y_valid)):
#         # прогноз лог-цены на 1 шаг
#         fc_log = arima_model.predict(n_periods=1)[0]
#         # обратно в цены
#         fc_price = float(np.exp(fc_log) - eps)
#         y_pred_valid.append(fc_price)
#         # обновляем модель фактическим лог-значением текущего дня валидации
#         current_real_adjclose = float(series.iloc[valid_idx[t]])
#         arima_model.update([np.log(current_real_adjclose + eps)])

#     y_pred_valid = np.array(y_pred_valid, dtype=float)

    
#     #  Метрики
#     rmse, mae, mape, r2 = metrics(ticker, y_true=y_valid, y_pred=np.array(y_pred_valid), model_name=model_name, model_dir=model_dir)


#     # Обучим финальную модель на всём ряде (на лог-уровне) для продакшн-прогноза
#     y_log_full = np.log(series.values + eps)
#     final_model = auto_arima(
#         y_log_full,
#         seasonal=seasonal,
#         m=m,
#         information_criterion="aic",
#         stepwise=True,
#         suppress_warnings=True,
#         error_action="ignore",
#         max_p=max_p, max_q=max_q,
#         max_P=max_P, max_Q=max_Q,
#         max_order=None,
#         trace=False,
#         with_intercept=True,
#         trend="t",
#         allowdrift=allow_drift
#     )

#     # Сохранение артефакта
#     artifact = {
#         "model": final_model,
#         "transform": {"type": "log", "epsilon": eps},
#         "y_last": float(series.values[-1]),
#         "y_log_last": float(y_log_full[-1]),
#         "last_date": str(pd.to_datetime(dates).iloc[-1].date()) if len(dates) else None,
#         "ticker": ticker,
#         "model_name": model_name,
#         "val_size": val_size
#     }

#     model_path = model_dir / f"{ticker}_{model_name}.joblib"
#     joblib.dump(artifact, model_path)

#     # results.json: добавим как метрики AIC/BIC (для информации), так и вал-метрики для сравнения
#     results_json_path = model_dir / f"{ticker}_{model_name}_results.json"
#     payload = {}
#     if results_json_path.exists():
#         try:
#             payload = json.loads(Path(results_json_path).read_text(encoding="utf-8"))
#         except Exception:
#             payload = {}

#     payload["metrics"] = {
#         "rmse": rmse, "mae": mae, "mape": mape, "r2": r2
#     }
#     # Информативные параметры модели (с треня на всем ряде)
#     payload["model_info"] = {
#         "order": tuple(int(x) for x in final_model.order),
#         "seasonal_order": tuple(int(x) for x in final_model.seasonal_order) if seasonal else (0,0,0,0),
#         "aic": float(final_model.aic()),
#         "bic": float(final_model.bic()),
#         "trend": "t",
#         "allow_drift": allow_drift,
#         "m": m
#     }
#     payload["val_size"] = int(val_size)
#     payload["feature_count"] = 1
#     payload["cv"] = None

#     with open(results_json_path, "w", encoding="utf-8") as f:
#         json.dump(payload, f, ensure_ascii=False, indent=2)

#     # Возвращаем вал-метрики, как требуется для сравнения с другими моделями
#     return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}, str(model_path)

# def train_arima_on_history_log(
#     df: pd.DataFrame,
#     ticker: str,
#     model_dir: str | Path = "result/models",
#     model_name: str = "ARIMA",
#     seasonal: bool = True,
#     m: int = 5,  # недельная псевдо-сезонность для б/дней
#     allow_drift: bool = True,
#     max_p: int = 7, max_q: int = 7, max_P: int = 3, max_Q: int = 3, # Увеличенные максимальные порядки
#     start_p: int = 1, start_q: int = 1, start_d: int = 1, # Задаем начальные порядки
#     start_P: int = 1, start_Q: int = 1, start_D: int = 0, # Задаем начальные сезонные порядки
#     val_size: int = VAL_SIZE,
#     exog_df: pd.DataFrame = None # Новый параметр для внешних регрессоров
# ):
#     model_dir = Path(model_dir)
#     model_dir.mkdir(parents=True, exist_ok=True)

#     # Источник ряда
#     series = df["AdjClose"].astype(float)
#     dates = pd.to_datetime(df["Date"]) if "Date" in df.columns else pd.bdate_range(periods=len(df), end=pd.Timestamp.today().normalize())

#     # Разделение на train/valid (валидация только на хвосте)
#     n = len(series)
#     val_size = int(val_size) if val_size is not None else 0
#     valid_start = max(0, n - val_size)
#     train_series = series.iloc[:valid_start].copy()
#     y_valid = series.iloc[valid_start:].dropna().values
#     valid_idx = np.arange(valid_start, valid_start + len(y_valid))

#     # Лог-преобразование
#     eps = 1e-9
#     y_log_train = np.log(train_series.values + eps)

#     # Подготовка exog-данных
#     exog_train = exog_df.iloc[:valid_start] if exog_df is not None else None
#     exog_valid = exog_df.iloc[valid_start:] if exog_df is not None else None
    
#     # Автоподбор ARIMA на лог-уровне треня
#     arima_model = auto_arima(
#         y_log_train,
#         X=exog_train, # Передаем exog для обучения
#         seasonal=seasonal,
#         m=m,
#         information_criterion="aic",
#         stepwise=False, # Изменено на False
#         suppress_warnings=True,
#         error_action="ignore",
#         max_p=max_p, max_q=max_q, max_d=1, # Ограничиваем d=1 для лог-доходностей
#         max_P=max_P, max_Q=max_Q, max_D=1 if seasonal else 0, # Ограничиваем D=1 для сезонных лог-доходностей
#         start_p=start_p, start_q=start_q, start_d=start_d, # Задаем начальные порядки
#         start_P=start_P, start_Q=start_Q, start_D=start_D, # Задаем начальные сезонные порядки
#         max_order=None,
#         trace=False,
#         with_intercept=True,
#         trend="c", # Изменено на "c" для постоянного дрейфа лог-доходностей
#         allowdrift=allow_drift
#     )

#     # Walk-forward на валидации: предсказываем лог-уровень на 1 шаг вперёд, обновляем модель фактом (на лог-уровне)
#     y_pred_valid = []
#     for t in range(len(y_valid)):
#         # прогноз лог-цены на 1 шаг
#         # Учитываем exog для прогноза, если он есть
#         fc_log = arima_model.predict(n_periods=1, X=exog_valid.iloc[t:t+1] if exog_valid is not None else None)[0]
#         # обратно в цены
#         fc_price = float(np.exp(fc_log) - eps)
#         y_pred_valid.append(fc_price)
        
#         # Обновляем модель фактическим лог-значением текущего дня валидации
#         current_real_adjclose = float(series.iloc[valid_idx[t]])
#         # Учитываем exog для обновления, если он есть
#         arima_model.update([np.log(current_real_adjclose + eps)], X=exog_valid.iloc[t:t+1] if exog_valid is not None else None)

#     y_pred_valid = np.array(y_pred_valid, dtype=float)

    
#     # Метрики
#     rmse, mae, mape, r2 = metrics(ticker, y_true=y_valid, y_pred=np.array(y_pred_valid), model_name=model_name, model_dir=model_dir)

#     # Обучим финальную модель на всём ряде (на лог-уровне) для продакшн-прогноза
#     y_log_full = np.log(series.values + eps)
#     final_model = auto_arima(
#         y_log_full,
#         X=exog_df, # Передаем полный exog для финальной модели
#         seasonal=seasonal,
#         m=m,
#         information_criterion="aic",
#         stepwise=False, # Изменено на False
#         suppress_warnings=True,
#         error_action="ignore",
#         max_p=max_p, max_q=max_q, max_d=1, # Ограничиваем d=1
#         max_P=max_P, max_Q=max_Q, max_D=1 if seasonal else 0, # Ограничиваем D=1
#         start_p=start_p, start_q=start_q, start_d=start_d, # Задаем начальные порядки
#         start_P=start_P, start_Q=start_Q, start_D=start_D, # Задаем начальные сезонные порядки
#         max_order=None,
#         trace=False,
#         with_intercept=True,
#         trend="c", # Изменено на "c"
#         allowdrift=allow_drift
#     )

#     # Сохранение артефакта
#     artifact = {
#         "model": final_model,
#         "transform": {"type": "log", "epsilon": eps},
#         "y_last": float(series.values[-1]),
#         "y_log_last": float(y_log_full[-1]),
#         "last_date": str(pd.to_datetime(dates).iloc[-1].date()) if len(dates) else None,
#         "ticker": ticker,
#         "model_name": model_name,
#         "val_size": val_size,
#         "exog_cols": list(exog_df.columns) if exog_df is not None else None # Сохраняем имена колонок exog
#     }

#     model_path = model_dir / f"{ticker}_{model_name}.joblib"
#     joblib.dump(artifact, model_path)

#     # results.json: добавим как метрики AIC/BIC (для информации), так и вал-метрики для сравнения
#     results_json_path = model_dir / f"{ticker}_{model_name}_results.json"
#     payload = {}
#     if results_json_path.exists():
#         try:
#             payload = json.loads(Path(results_json_path).read_text(encoding="utf-8"))
#         except Exception:
#             payload = {}

#     payload["metrics"] = {
#         "rmse": rmse, "mae": mae, "mape": mape, "r2": r2
#     }
#     # Информативные параметры модели (с треня на всем ряде)
#     payload["model_info"] = {
#         "order": tuple(int(x) for x in final_model.order),
#         "seasonal_order": tuple(int(x) for x in final_model.seasonal_order) if seasonal else (0,0,0,0),
#         "aic": float(final_model.aic()),
#         "bic": float(final_model.bic()),
#         "trend": "c", # Изменено на "c"
#         "allow_drift": allow_drift,
#         "m": m
#     }
#     payload["val_size"] = int(val_size)
#     payload["feature_count"] = 1 + (len(exog_df.columns) if exog_df is not None else 0)
#     payload["cv"] = None

#     with open(results_json_path, "w", encoding="utf-8") as f:
#         json.dump(payload, f, ensure_ascii=False, indent=2)

#     # Возвращаем вал-метрики, как требуется для сравнения с другими моделями
#     return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}, str(model_path)


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
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}, str(model_path)