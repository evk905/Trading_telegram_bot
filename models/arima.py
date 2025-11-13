import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

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


