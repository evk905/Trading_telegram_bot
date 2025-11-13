import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Keras LSTM
from tensorflow import keras
from tensorflow.keras import layers

from config import  VAL_SIZE
from metrics_plots import metrics
from models.rf import make_features, time_split_train_valid



def build_lstm_model(input_dim: int, units: int = 64, dropout: float = 0.1) -> keras.Model:
    """
    Простой MLP/LSTM для табличных признаков:
    - Если хотим настоящую LSTM по временным окнам, нужен 3D-тензор (samples, time, features).
      Ниже — упрощенный вариант «одношаговый» через Dense (быстрее и стабильнее).
      Если всё-таки LSTM: добавьте формирование окон и замените Dense-сеть на LSTM.
    """
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = layers.Dense(units, activation="relu")(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(units // 2, activation="relu")(x)
    outputs = layers.Dense(1, name="y")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def train_lstm_on_history(
    df: pd.DataFrame,
    ticker: str,
    model_dir: str | Path = "result/models",
    model_name: str = "LSTM",
    val_size: int = VAL_SIZE,
    epochs: int = 60,
    batch_size: int = 32,
    early_stopping_patience: int = 8,
    standardize: bool = True
):
    """
    Обучает простую нейросеть (Dense/LSTM-лайт) на тех же признаках, что и RandomForest.
    Цель: AdjClose[t+1]. Разбиение по времени такое же, как в RF.
    Возвращает: (dict метрик, путь к файлу модели).
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Данные и признаки те же, что в RF 
    feat_df = make_features(df)
    y = feat_df["y"].astype(float)
    feature_cols = [c for c in feat_df.columns if c not in ["Date", "AdjClose", "y"]]
    X = feat_df[feature_cols].astype(float)

    # Временной split
    train_idx, valid_idx = time_split_train_valid(X, y, val_size=val_size)
    X_train, y_train = X.iloc[train_idx].values, y.iloc[train_idx].values
    X_valid, y_valid = X.iloc[valid_idx].values, y.iloc[valid_idx].values

    # Стандартизация фич (по train), цель не масштабируем (так честнее для RMSE/MAE)
    scaler = None
    if standardize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)

    # Модель
    model = build_lstm_model(input_dim=X_train.shape[1], units=64, dropout=0.1)

    # Обучение с ранней остановкой по валидации (MSE)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
        )
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
        shuffle=False  # важно для временных рядов
    )

    # Предсказание и метрики
    y_pred = model.predict(X_valid, verbose=0).reshape(-1)
    rmse, mae, mape, r2 = metrics(ticker, y_true=y_valid, y_pred=y_pred, model_name=model_name, model_dir=model_dir)

    # Сохранение модели: keras-сеть + препроцессор
    # Сохраним в joblib словарь с сериализованной Keras-моделью в формате SavedModel + scaler
    # Саму Keras-модель сохраним в подкаталог, а в joblib — метаданные + путь
    keras_path = Path(model_dir) / f"{ticker}_{model_name}.keras"
    model.save(keras_path, include_optimizer=True)

    payload = {
        "keras_path": str(keras_path),
        "feature_cols": feature_cols,
        "target": "y",
        "ticker": ticker,
        "model_name": model_name,
        "val_size": val_size,
        "train_rows": int(len(train_idx)),
        "valid_rows": int(len(valid_idx)),
        "last_date": str(df["Date"].max().date()) if "Date" in df.columns else None,
        "scaler": scaler,  # может быть None
        "note": "Dense NN на табличных признаках; прогноз AdjClose[t+1]"
    }
    model_path = Path(model_dir) / f"{ticker}_{model_name}.joblib"
    joblib.dump(payload, model_path)

    # JSON для сравнения
    results_json = {
        "ticker": ticker,
        "model_name": model_name,
        "metrics": {"rmse": float(rmse), "mae": float(mae), "mape": float(mape), "r2": float(r2)},
        "val_size": int(val_size),
        "feature_count": len(feature_cols),
        "cv": None
    }
    with open(Path(model_dir) / f"{ticker}_{model_name}_results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    return results_json["metrics"], str(model_path)