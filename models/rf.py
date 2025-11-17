import os
import sys
import json
import joblib
import typing as t
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from config import FORECAST_HORIZON, VAL_SIZE, model_name
from service.metrics_plots import metrics



def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Формирование признаков для модели:
    - лагаемые значения AdjClose
    - скользящие средние/стд
    - дневные доходности и их статистики
    - волатильность по High/Low
    Цель: AdjClose на следующий день.
    """
    out = df.copy()
    # Базовые лаги по цене закрытия (adj)
    for lag in [1, 2, 3, 5, 10]:
        out[f"AdjClose_lag{lag}"] = out["AdjClose"].shift(lag)

    # Доходность
    out["ret_1"] = out["AdjClose"].pct_change()
    
    # Скользящие статистики по цене
    for w in [3, 5, 10, 20]:
        out[f"sma_{w}"] = out["AdjClose"].rolling(w).mean()
        out[f"std_{w}"] = out["AdjClose"].rolling(w).std()
        out[f"ret_mean_{w}"] = out["ret_1"].rolling(w).mean()
        out[f"ret_std_{w}"] = out["ret_1"].rolling(w).std()

    # Диапазон дня и относительная волатильность
    out["hl_range"] = (out["High"] - out["Low"]) / out["Open"].replace(0, np.nan)
    out["oc_change"] = (out["Close"] - out["Open"]) / out["Open"].replace(0, np.nan)

    # Объемы и их статистики
    out["vol_chg"] = out["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    for w in [5, 20]:
        out[f"vol_sma_{w}"] = out["Volume"].rolling(w).mean()
        out[f"vol_std_{w}"] = out["Volume"].rolling(w).std()

    # Целевой сдвиг: прогноз следующего дня AdjClose
    out["y"] = out["AdjClose"].shift(-1)

    # Удалим строки с NaN, образовавшимися от лагов/rolling и последнюю строку (y=NaN)
    out = out.dropna().reset_index(drop=True)
    return out

def time_split_train_valid(X: pd.DataFrame, y: pd.Series, val_size: int = VAL_SIZE):
    """
    Временной split: обучаем на начальном отрезке, валидируем на последних val_size наблюдениях.
    """
    if len(X) <= val_size + 30:
        # минимально оставим хотя бы 30 точек на обучение
        raise ValueError(f"Слишком мало данных для разбиения: {len(X)} строк при val_size={val_size}")
    train_idx = np.arange(0, len(X) - val_size)
    valid_idx = np.arange(len(X) - val_size, len(X))
    return train_idx, valid_idx

def rf_default_params() -> dict:
    return {
        "n_estimators": 600,
        "max_depth": None,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": 42,
        "oob_score": False,
        "bootstrap": True,
    }

def train_random_forest_on_history(
    df: pd.DataFrame,
    ticker: str,
    model_dir: t.Union[str, Path] = "result/models",
    model_name: str = "RandomForest",
    val_size: int = VAL_SIZE,
    params: t.Optional[dict] = None,
    do_timeseries_cv: bool = False,
    cv_splits: int = 3
) -> t.Tuple[dict, str]:
    """
    Обучает RandomForest на исторических данных тикера, считает метрики на тесте (валидации)
    и сохраняет модель в файл. Возвращает:
      - словарь метрик {'rmse':..., 'mae':..., 'mape':..., 'r2':..., 'val_size':...}
      - путь к файлу модели (joblib)
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Построение признаков
    feat_df = make_features(df)
    y = feat_df["y"].astype(float)
    feature_cols = [c for c in feat_df.columns if c not in ["Date", "AdjClose", "y"]]
    X = feat_df[feature_cols].astype(float)

    # Временной split
    train_idx, valid_idx = time_split_train_valid(X, y, val_size=val_size)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # (Опционально) кросс-валидация по времени для оценки стабильности
    cv_scores = None
    if do_timeseries_cv:
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        cv_rmses = []
        for tr_idx, te_idx in tscv.split(X_train):
            X_tr, X_te = X_train.iloc[tr_idx], X_train.iloc[te_idx]
            y_tr, y_te = y_train.iloc[tr_idx], y_train.iloc[te_idx]
            rf_cv = RandomForestRegressor(**(params or rf_default_params()))
            rf_cv.fit(X_tr, y_tr)
            pred_te = rf_cv.predict(X_te)
            rmse_cv = float(np.sqrt(mean_squared_error(y_te, pred_te)))
            cv_rmses.append(rmse_cv)
        cv_scores = {
            "rmse_mean": float(np.mean(cv_rmses)),
            "rmse_std": float(np.std(cv_rmses)),
            "splits": cv_splits
        }

    # Обучение финальной модели на train
    rf = RandomForestRegressor(**(params or rf_default_params()))
    rf.fit(X_train, y_train)

    # Предсказание на валидации и метрики
    y_pred = rf.predict(X_valid)
    rmse, mae, mape, r2 = metrics(
        ticker, 
        y_true=y_valid.values, 
        y_pred=y_pred, 
        model_name=model_name, 
        model_dir=model_dir
        )

    # Сохранение модели и вспомогательных артефактов
    model_path = model_dir / f"{ticker}_{model_name}.joblib"
    joblib.dump(
        {
            "model": rf,
            "feature_cols": feature_cols,
            "target": "y",
            "ticker": ticker,
            "model_name": model_name,
            "val_size": val_size,
            "train_rows": int(len(X_train)),
            "valid_rows": int(len(X_valid)),
            "fitted_on_last_date": str(df["Date"].max().date()) if "Date" in df.columns else None,
        },
        model_path
    )

    # Сохранение метаданных/результатов для удобства сравнения моделей
    results_json = {
        "ticker": ticker,
        "model_name": model_name,
        "metrics": {"rmse": float(rmse), "mae": float(mae), "mape": float(mape), "r2": float(r2)},
        "val_size": int(val_size),
        "feature_count": len(feature_cols),
        "cv": cv_scores
    }
    with open(model_dir / f"{ticker}_{model_name}_results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    return results_json["metrics"], str(model_path)
