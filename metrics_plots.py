import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import FORECAST_HORIZON, VAL_SIZE


# Расчет и сохранение метрик
def metrics(
        ticker, 
        y_true, 
        y_pred, 
        model_dir,
        model_name, 
        val_size=VAL_SIZE
        ):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100
    r2 = r2_score(y_true, y_pred)
    out_metrics = model_dir / f"{ticker}_{model_name}_metrics.txt"
    with open(out_metrics, "w", encoding="utf-8") as f:
        f.write(f"Метрики на валидации (последние {val_size} дней):\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE:  {mae:.4f}\n")
        f.write(f"MAPE: {mape:.2f}%\n")
        f.write(f"R2:   {r2:.4f}\n")

    return rmse, mae, mape, r2

# Построить и сохранить график
def plot_and_save(
        ticker, 
        dates, 
        y_hist, 
        pred_dates, 
        y_pred, 
        model_name,
        out_dir="outputs", 
        forecast_horizon=None, 
        dpi=150, 
        return_abs=True):
    """
    Строит и сохранят график прогноза. Возвращает путь к сохранённому файлу.
    - ticker: str
    - dates: iterable дат для исторических значений
    - y_hist: iterable значений истории
    - pred_dates: iterable дат прогноза
    - y_pred: iterable значений прогноза
    - model_name: имя модели для файла/заголовка
    - out_dir: каталог для сохранения
    - forecast_horizon: если None, берём глобальный FORECAST_HORIZON; иначе используем переданное значение
    - dpi: качество сохранения
    - return_abs: вернуть абсолютный путь (True) или относительный (False)
    """
    fh = forecast_horizon if forecast_horizon is not None else FORECAST_HORIZON

    # Безопасное имя файла (добавим дату/время, чтобы не перезаписывать)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_{model_name}_forecast_{fh}d_{ts}.png"

    # Убедимся, что каталог существует
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
   
    title = f"{ticker}: {model_name} прогноз на {fh} дней"

    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_hist, label="История (AdjClose)", linewidth=1.7)
    plt.plot(pred_dates, y_pred, label=f"Прогноз +{len(y_pred)} дн", linewidth=1.7,
             linestyle="--", color="tomato")
    plt.xlabel("Дата")
    plt.ylabel("Цена")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

    return os.path.abspath(out_path) if return_abs else out_path