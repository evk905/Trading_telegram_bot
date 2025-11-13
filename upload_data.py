import sys
import datetime as dt
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd


try:
    import yfinance as yf
except ImportError:
    print("Не найден модуль yfinance. Установите: pip install yfinance", file=sys.stderr)
    sys.exit(1)

# Ввод пользователя
def prompt_user() -> Tuple[str, float]:
    ticker = input("Введите тикер (например, AAPL): ").strip().upper()
    if not ticker:
        print("Тикер не может быть пустым.", file=sys.stderr)
        sys.exit(1)

    amount_str = input("Введите сумму инвестиций (в валюте тикера): ").strip().replace(",", ".")
    try:
        amount = float(amount_str)
        if amount <= 0:
            raise ValueError
    except ValueError:
        print("Некорректная сумма. Введите положительное число.", file=sys.stderr)
        sys.exit(1)

    return ticker, amount

# Загрузка  ежедневных  цен за последние 2 года
def load_prices(ticker: str) -> pd.DataFrame:
    end = dt.datetime.now().date()
    start = end - dt.timedelta(days=365 * 2 + 7)  # небольшой запас на нерабочие дни
    
    # Интервал '1d' для дневных свечей
    data = yf.download(
        ticker, 
        start=start.isoformat(), 
        end=end.isoformat(),
        interval="1d", 
        auto_adjust=False, 
        progress=False
        )
    
    if data is None or data.empty:
        print(f"Не удалось получить данные для тикера {ticker}. Проверьте тикер или соединение.", file=sys.stderr)
        sys.exit(1)
    
    # Убедимся, что индекс — это столбец Date
    data = data.reset_index()

    # Приведём названия столбцов к простому виду
    data.columns = [c if not isinstance(c, tuple) else "_".join([x for x in c if x]) for c in data.columns]
    
    # Ожидаемые колонки и карта переименования
    rename_map = {
        f"Adj Close_{ticker}": "AdjClose",
        f"Close_{ticker}": "Close",
        f"High_{ticker}": "High",
        f"Low_{ticker}": "Low",
        f"Open_{ticker}": "Open",
        f"Volume_{ticker}": "Volume",
        }

    # Проверим наличие всех ключей для переименования
    missing = [col for col in rename_map.keys() if col not in data.columns]
    if missing:
        raise KeyError(f"В CSV отсутствуют ожидаемые колонки: {missing}")

    # Попробуем переименовать; если что-то пойдет не так — пробросим осмысленную ошибку
    try:
        data = data.rename(columns=rename_map)
    except Exception as e:
        raise RuntimeError(f"Не удалось переименовать колонки с тикером {ticker}: {e}")

    # Удалим дубликаты по дате, оставим последнюю запись
    data = data.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)


    # Приведение типа даты и сортировка
    data["Date"] = pd.to_datetime(data["Date"], utc=False)
    data = data.sort_values("Date").reset_index(drop=True)

    return data


# Функция сохранение данных в файл
def save_csv(df, ticker: str, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / f"{ticker}_2y_daily.csv"
    df.to_csv(out_path, index=False)
    return out_path

















# # Функция возвращает последнюю цену и количество акций для покупки
# def compute_shares_info(df, ticker: str, amount: float):
#     # Берём последнюю доступную цену закрытия
#     last_row = df.sort_values("Date").iloc[-1]
#     # close_col_candidates = ["Close", "Adj Close", "Close_0", "Adj Close_0"]
#     close_col_candidates = [f"Adj Close_{ticker}",]
#     close_price = None
#     for c in close_col_candidates:
#         if c in df.columns:
#             close_price = float(last_row[c])
#             break
#     if close_price is None:
#         # Иногда yfinance возвращает мультииндексные названия — уже упрощены выше
#         raise RuntimeError("Не удалось определить столбец с ценой закрытия.")
#     shares = amount / close_price if close_price > 0 else 0.0
#     return close_price, shares


# Ссхранение расчета в файл
# def save_summary(ticker: str, amount: float, close_price: float, shares: float, folder: Path):
#     folder.mkdir(parents=True, exist_ok=True)
#     summary_path = folder / f"{ticker}_summary.txt"
#     with open(summary_path, "w", encoding="utf-8") as f:
#         f.write(f"Тикер: {ticker}\n")
#         f.write(f"Сумма инвестиций: {amount:.2f}\n")
#         f.write(f"Последняя цена закрытия: {close_price:.4f}\n")
#         f.write(f"Ориентировочно акций на сумму: {shares:.6f}\n")
#     return summary_path


# def main():
#     ticker, amount = prompt_user()
#     print(f"Загружаю котировки {ticker} за ~последние 2 года...")
#     df = load_prices(ticker)
#     out_folder = Path("data")
#     csv_path = save_csv(df, ticker, out_folder)
#     print(f"Готово. Данные сохранены: {csv_path}")
#     # try:
#     #     close_price, shares = compute_shares_info(df, ticker, amount)
#     #     summary_path = save_summary(ticker, amount, close_price, shares, out_folder)
#     #     print(f"Готово. Данные сохранены: {csv_path}")
#     #     print(f"Сводка сохранена: {summary_path}")
#     #     print(f"Последняя цена закрытия: {close_price:.4f}. Можно купить примерно {shares:.6f} акций на сумму {amount:.2f}.")
#     # except Exception as e:
#     #     print(f"CSV сохранён: {csv_path}, но не удалось создать сводку: {e}", file=sys.stderr)


# if __name__ == "__main__":
#     main()