import logging
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from aiogram import Bot, Dispatcher, types
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

from service.upload_data import load_prices
from service.select_best_models import select_best_model, forecast_bot
from config import FORECAST_HORIZON, VAL_SIZE



# Токен вашего бота
# API_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
API_TOKEN = '8547584856:AAEYbRCIGyGq--CMNoMIzxIilnFkc7h3Mf0'

# Настройка логирования
logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

class InvestmentForm(StatesGroup):
    ticker = State()
    amount = State()



@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    keyboard = InlineKeyboardMarkup()
    start_button = InlineKeyboardButton("Начать", callback_data='start_investment')
    keyboard.add(start_button)
    await message.reply("Привет! Я бот для прогноза акций. Нажми 'Начать', чтобы ввести тикер и сумму.", reply_markup=keyboard)

@dp.callback_query_handler(lambda c: c.data == 'start_investment')
async def process_start(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    await InvestmentForm.ticker.set()
    await bot.send_message(callback_query.from_user.id, "Введите тикер компании (например, AAPL, MSFT):")

@dp.message_handler(state=InvestmentForm.ticker)
async def process_ticker(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['ticker'] = message.text.upper()
    await InvestmentForm.next()
    await message.reply("Введите сумму для условной инвестиции (в USD):")

@dp.message_handler(state=InvestmentForm.amount)
async def process_amount(message: types.Message, state: FSMContext):
    try:
        amount = float(message.text)
    except ValueError:
        await message.reply("Пожалуйста, введите числовое значение для суммы.")
        return
    
    async with state.proxy() as data:
        ticker = data['ticker']
    
    await state.finish()
    
    # Выполнение логики
    df = load_prices(ticker)
     # Приведение типа даты и сортировка
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    best_model, best_path = select_best_model(df, ticker)
    forecast_info, summary, recommendations, trades, change_text, plot_path, summary_text = forecast_bot(df, ticker, amount, best_model, best_path)
    
    # Отправка результатов
    await message.reply(f"Прогноз для {ticker}:\n{summary_text}")
    await message.reply(change_text)
    
    # Отправка графика
    with open(plot_path, 'rb') as photo:
        await bot.send_photo(message.chat.id, photo)
    
    # Рекомендации
    rec_text = "Рекомендации:\n" + "\n".join(recommendations)
    await message.reply(rec_text)
    
    # Торговли
    if trades:
        trades_text = "Симуляция торгов:\n" + "\n".join(trades)
        await message.reply(trades_text)
    
    # Сводка
    await message.reply("Сводка:\n" + summary["strategy"] + "\n" + summary["estimated_profit"])

if __name__ == '__main__':
    from aiogram import executor
    executor.start_polling(dp, skip_updates=True)