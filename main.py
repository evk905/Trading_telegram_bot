
from pathlib import Path
import pandas as pd
from upload_data import prompt_user, load_prices, save_csv
from config import FORECAST_HORIZON, VAL_SIZE
from models.rf import train_random_forest_on_history
from models.rf_pred import rf_forecast_30_days
from models.arima import  train_arima_on_history_log
from models.lstm import train_lstm_on_history
from models.arima_pred import forecast_30_days_arima_log 


def main():
    ticker, amount = prompt_user()
    print(f"Загружаю котировки {ticker} за ~последние 2 года...")
    
    df = load_prices(ticker)
    out_folder = Path("data")
    csv_path = save_csv(df, ticker, out_folder)
    print(f"Готово. Данные сохранены: {csv_path}") # data/{ticker}_2y_daily.csv

   # Приведение типа даты и сортировка
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

#    # Обучение RF
#     metrics_rf, path_rf = train_random_forest_on_history(df, ticker)
#     print(metrics_rf, path_rf)

    # Обучение ARIMA
    metrics_arima, path_arima = train_arima_on_history_log(df, ticker)
    print(metrics_arima, path_arima) 

    # # Обучение LSTM
    # metrics_lstm, path_lstm = train_lstm_on_history(df, ticker)
    # print(metrics_lstm, path_lstm)


    # # RF Прогноз на 30 дней, график и запись в JSON
    # rf_forecast_info = rf_forecast_30_days(
    #     df,
    #     ticker,
    #     model_artifact_path=path_rf
    # )
    # print(rf_forecast_info["summary"])
    # print("График сохранен в:", rf_forecast_info["plot_path"])
    
 
    # ARIMA Форвард-прогноз на 30 дней, график и запись в JSON
    arima_forecast_info = forecast_30_days_arima_log (
        df,
        ticker,
        model_artifact_path=path_arima
    )
    print(arima_forecast_info["summary"])
    print("График сохранен в:", arima_forecast_info["plot_path"])


if __name__ == "__main__":
    main()
