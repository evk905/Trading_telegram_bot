import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import joblib
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from config import FORECAST_HORIZON, VAL_SIZE
from service.metrics_plots import metrics, plot_and_save

import matplotlib.pyplot as plt
from datetime import timedelta

def train_lstm_on_history(
    df: pd.DataFrame,
    ticker: str,
    model_dir: str | Path = "result/models",
    model_name: str = "LSTM",
    val_size: int = VAL_SIZE,
    epochs: int = 100,
    batch_size: int = 32,
    early_stopping_patience: int = 10,
    standardize: bool = True,
    window_size: int = 20,
):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = ['AdjClose']  # Унивариантный режим, только AdjClose
    target_col = 'AdjClose'
    target_idx = 0  # Единственная фича

    # Разделение данных
    n = len(df)
    train_df = df.iloc[:n - val_size]
    valid_df = df.iloc[n - val_size:]
    train_idx = train_df.index
    valid_idx = valid_df.index

    # Подготовка данных
    train_data = train_df[feature_cols].values  # (n_train, 1)
    valid_data = valid_df[feature_cols].values  # (n_valid, 1)

    if standardize:
        scaler = MinMaxScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        valid_data = scaler.transform(valid_data)
    else:
        scaler = None

    # Функция для создания последовательностей
    def create_sequences(data, window_size, target_idx):
        X, y = [], []
        for i in range(len(data) - window_size):
            seq = data[i:i + window_size]
            label = data[i + window_size, target_idx]
            X.append(seq)
            y.append(label)
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data, window_size, target_idx)
    X_valid, y_valid = create_sequences(valid_data, window_size, target_idx)

    # Построение модели LSTM (унивариантная)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size, 1)))
    model.add(LSTM(units=25, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Early stopping
    es = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=1, restore_best_weights=True)

    # Обучение модели
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid), callbacks=[es])

    # Предсказание и метрики
    y_pred = model.predict(X_valid).flatten()

    # Обратное преобразование, если была стандартизация
    if standardize:
        data_range = scaler.data_max_[target_idx] - scaler.data_min_[target_idx]
        y_valid_inv = y_valid * data_range + scaler.data_min_[target_idx]
        y_pred_inv = y_pred * data_range + scaler.data_min_[target_idx]
    else:
        y_valid_inv = y_valid
        y_pred_inv = y_pred

    rmse, mae, mape, r2 = metrics(
        ticker,
        y_true=y_valid_inv,
        y_pred=y_pred_inv,
        model_name=model_name,
        model_dir=model_dir,
        val_size=len(y_valid_inv) 
    )
    # Сохранение модели
    keras_path = Path(model_dir) / f"{ticker}_{model_name}.keras"
    model.save(keras_path, include_optimizer=True)

    payload = {
        "keras_path": str(keras_path),
        "feature_cols": feature_cols,
        "target": "y",
        "ticker": ticker,
        "model_name": model_name,
        "val_size": val_size,
        "window_size": window_size,
        "lookback": window_size,  # Добавлено для совместимости с функцией прогноза
        "train_rows": int(len(train_idx)),
        "valid_rows": int(len(valid_idx)),
        "last_date": str(df["Date"].max().date()) if "Date" in df.columns else None,
        "scaler": scaler,
        "note": "LSTM по временным окнам; прогноз AdjClose[t+1]"
    }

    model_path = Path(model_dir) / f"{ticker}_{model_name}.joblib"
    joblib.dump(payload, model_path)

    # JSON для сравнения
    results_json = {
        "ticker": ticker,
        "model_name": model_name,
        "metrics": {
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "r2": float(r2)
        },
        "val_size": int(val_size),
        "feature_count": len(feature_cols),
        "window_size": window_size,
        "cv": None
    }
    with open(Path(model_dir) / f"{ticker}_{model_name}_results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    
    
    return results_json["metrics"], str(model_path)

