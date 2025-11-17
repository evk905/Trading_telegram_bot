from datetime import datetime
import os
import logging
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import asyncio

from aiogram import Bot, Dispatcher, Router, types, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, FSInputFile

from service.upload_data import load_prices
from service.select_best_models import select_best_model, forecast_bot
from config import FORECAST_HORIZON, VAL_SIZE


LOG_FILE = os.path.join("logs", "logs.txt")
os.makedirs("logs", exist_ok=True)


def log_request(
    user_id: int,
    ticker: str,
    amount: float,
    best_model_name: str,
    metric_value: float,
    estimated_profit: float,
) -> None:
    """
    Логирует один запрос пользователя в текстовый файл LOG_FILE.
    Формат строки: user_id | datetime | ticker | amount | model | metric | profit
    """
    # Текущие дата и время в удобном формате
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Готовим строку лога
    line = (
        f"{user_id} | {timestamp} | {ticker} | {amount} | "
        f"{best_model_name} | {metric_value} | {estimated_profit}\n"
    )

    # Гарантируем, что файл будет создан, если его нет
    # и запись будет добавлена в конец
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)

# Токен бота
API_TOKEN = '8547584856:AAEYbRCIGyGq--CMNoMIzxIilnFkc7h3Mf0'

logging.basicConfig(level=logging.INFO)

router = Router()


class InvestmentForm(StatesGroup):
    ticker = State()
    amount = State()


@router.message(Command(commands=["start"]))
async def send_welcome(message: types.Message):
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Начать", callback_data="start_investment")]
        ]
    )
    await message.reply(
        "Привет! Я бот для прогноза акций. Нажми 'Начать', чтобы ввести тикер и сумму.",
        reply_markup=keyboard,
    )


@router.callback_query(F.data == "start_investment")
async def process_start(callback_query: types.CallbackQuery, state: FSMContext, bot: Bot):
    await callback_query.answer()
    await state.set_state(InvestmentForm.ticker)
    await bot.send_message(
        callback_query.from_user.id,
        "Введите тикер компании (например, AAPL, MSFT):",
    )


@router.message(StateFilter(InvestmentForm.ticker))
async def process_ticker(message: types.Message, state: FSMContext):
    await state.update_data(ticker=message.text.upper())
    await state.set_state(InvestmentForm.amount)
    await message.reply("Введите сумму для условной инвестиции (в USD):")


@router.message(StateFilter(InvestmentForm.amount))
async def process_amount(message: types.Message, state: FSMContext, bot: Bot):
    try:
        amount = float(message.text)
    except ValueError:
        await message.reply("Пожалуйста, введите числовое значение для суммы.")
        return

    data = await state.get_data()
    ticker = data["ticker"]

    # очищаем состояние
    await state.clear()

    # Загрузка и подготовка данных
    df = load_prices(ticker)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Выбираем лучшую модель
    best_model, best_path, best_mae = select_best_model(df, ticker)
    
    # Строим прогноз и торговую стратегию
    (
        forecast_info,
        summary,
        recommendations,
        trades,
        plot_path,
        summary_text,
    ) = forecast_bot(df, ticker, amount, best_model, best_path)

    # Получаем числовое значение прибыли из summary
    estimated_profit_value = float(summary.get("estimated_profit_value", 0.0))

    # Имя/описание модели
    best_model_name = getattr(best_model, "name", str(best_model))

    # Логирование запроса
    user_id = message.from_user.id
    log_request(
        user_id=user_id,
        ticker=ticker,
        amount=amount,
        best_model_name=best_model_name,
        metric_value=float(best_mae),
        estimated_profit=float(estimated_profit_value),
        # при желании можно добавить:
        # summary_text=summary_text,
        # change_text=change_text,
        # results_json_path=forecast_info.get("results_json_path"),
    )

    # Отправка результатов
    await message.reply(f"Прогноз для {ticker}:\n{summary_text}")

    # Отправка графика
    await bot.send_photo(message.chat.id, photo=FSInputFile(plot_path))

    # Рекомендации
    rec_text = "Рекомендации:\n" + "\n".join(recommendations)
    await message.reply(rec_text)

    # Симуляция торгов
    if trades:
        trades_text = "Симуляция торгов:\n" + "\n".join(trades)
        await message.reply(trades_text)

    # Сводка
    await message.reply(
        "Сводка:\n" + summary["strategy"] + "\n" + summary["estimated_profit"]
    )


async def main():
    bot = Bot(token=API_TOKEN)
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)

    dp.include_router(router)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())