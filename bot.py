import asyncio
import logging
import os
import numpy as np
import cv2
import pickle
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from tensorflow.keras.models import load_model
from scipy.signal import find_peaks, resample
import matplotlib.pyplot as plt

API_TOKEN = '8309832107:AAHJzS8lD9ym4qwUWl8ZJo6kFD97g1Rjo6Q' 
MODEL_PATH = 'ecg_model.h5'
CLASSES_PATH = 'classes.pkl'

logging.basicConfig(level=logging.INFO)
bot = Bot(token=API_TOKEN)
dp = Dispatcher()

print("------------------------------------------------")
print("✅ БОТ ЗАПУЩЕНО: v14.0 (GRAND DIAGNOSTIC)")
print("------------------------------------------------")

# --- AI ---
ai_available = False
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
        model = load_model(MODEL_PATH)
        with open(CLASSES_PATH, 'rb') as f:
            classes_dict = pickle.load(f)
        ai_available = True
    else:
        print("⚠️ AI модель не знайдено.")
except Exception as e:
    print(f"Помилка AI: {e}")

class ECGState(StatesGroup):
    waiting_for_animal_type = State()
    waiting_for_weight = State()
    waiting_for_method = State()
    waiting_for_photo = State()
    waiting_for_duration = State()
    waiting_for_squares = State()

def get_animal_keyboard():
    buttons = [
        [InlineKeyboardButton(text="🐱 Кіт", callback_data="animal_cat")],
        [InlineKeyboardButton(text="🐶 Собака", callback_data="animal_dog")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_method_keyboard():
    buttons = [
        [InlineKeyboardButton(text="📸 По фото (Авто)", callback_data="method_photo")],
        [InlineKeyboardButton(text="📏 По клітинках (Точно)", callback_data="method_grid")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_speed_keyboard():
    buttons = [
        [InlineKeyboardButton(text="25 мм/с", callback_data="speed_25")],
        [InlineKeyboardButton(text="50 мм/с", callback_data="speed_50")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_reference_values(animal_type, weight):
    min_norm, max_norm = 0, 0
    if "cat" in animal_type:
        min_norm, max_norm = 120, 200 # Трохи звузили верхню межу для безпеки
    elif "dog" in animal_type:
        if weight < 5: min_norm, max_norm = 100, 160
        elif weight < 15: min_norm, max_norm = 80, 140
        elif weight < 30: min_norm, max_norm = 70, 120
        elif weight < 50: min_norm, max_norm = 60, 100
        else: min_norm, max_norm = 50, 90
    return min_norm, max_norm

def analyze_pathologies(signal, peaks, cv, ai_verdict):
    warnings = []
    suspicion_score = 0
    
    if len(peaks) < 3: return warnings, 0

    
    amplitudes = signal[peaks]
    amplitude_cv = np.std(amplitudes) / np.mean(amplitudes)
    
    if amplitude_cv > 0.15 and cv < 0.15:
        warnings.append("⚠️ **Електрична альтернація** (R-зубці різної висоти).\n  _👉 Виключіть випіт у перикард (тампонаду)._")
        suspicion_score += 2

    t_ratios = []
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i+1]
        margin = int((end - start) * 0.15)
        segment = signal[start+margin : end-margin]
        if len(segment) > 0:
            max_t = np.max(segment)
            max_r = signal[start]
            if max_r > 1: t_ratios.append(max_t / max_r)
    
    if t_ratios:
        avg_t_ratio = np.mean(t_ratios)
        if avg_t_ratio > 0.50:
            warnings.append("🧪 **ГІПЕРкаліємія** (Високі зубці T > 50% R).\n  _👉 Перевірте електроліти та сечовипускання._")
            suspicion_score += 2
        elif avg_t_ratio < 0.08:
            warnings.append("🧪 **ГІПОкаліємія** (Пласкі зубці T).\n  _👉 Можлива слабкість/блювота._")
            suspicion_score += 1

    
    rr_intervals = np.diff(peaks)
    if len(rr_intervals) > 3:
        is_bigeminy = True
        avg_diff = np.mean(rr_intervals)
        for i in range(len(rr_intervals) - 1):
            if abs(rr_intervals[i] - rr_intervals[i+1]) < avg_diff * 0.2:
                is_bigeminy = False
                break
        
        if is_bigeminy and cv > 0.2:
             warnings.append("❤️ **Бігемінія** (Чергування інтервалів).\n  _👉 Характерно для стійкої екстрасистолії._")
             suspicion_score += 2

    return warnings, suspicion_score

def get_full_diagnosis(bpm, min_norm, max_norm, cv, ai_verdict, animal_type, warnings):
    verdict_lines = []
    severity = "green"

    if bpm < min_norm:
        verdict_lines.append(f"• Частота: 🔴 **Брадикардія** ({bpm} < {min_norm})")
        severity = "red" if bpm < min_norm * 0.7 else "yellow"
    elif bpm > max_norm:
        verdict_lines.append(f"• Частота: 🔴 **Тахікардія** ({bpm} > {max_norm})")
        severity = "red" if bpm > max_norm * 1.2 else "yellow"
    else:
        verdict_lines.append("• Частота: ✅ Нормосистолія")

    if cv > 0.15:
        if "dog" in animal_type and not warnings and ai_verdict in ["Normal (N)", "Норма (N)"] and cv < 0.35:
            verdict_lines.append("• Ритм: ⚠️ Синусова аритмія (Ймовірно фізіологічна норма)")
        else:
            verdict_lines.append(f"• Ритм: ❌ **Нерегулярний** (CV {int(cv*100)}%)")
            if severity == "green": severity = "yellow"
    else:
        verdict_lines.append("• Ритм: ✅ Правильний")

    for w in warnings:
        verdict_lines.append(f"• {w}")
        if "ГІПЕР" in w or "Альтернація" in w: severity = "red"
        elif severity == "green": severity = "yellow"

    if "V" in ai_verdict or "VEB" in ai_verdict:
        verdict_lines.append("• Комплекси: 🔴 **Шлуночкові екстрасистоли (VPC)**")
        severity = "red"
    elif "S" in ai_verdict:
        verdict_lines.append("• Комплекси: 🟡 Надшлуночкові екстрасистоли")
    
    title = ""
    if severity == "green": title = "✅ КЛІНІЧНА НОРМА"
    elif severity == "yellow": title = "⚠️ ПОМІРНІ ВІДХИЛЕННЯ"
    else: title = "🚨 ВИРАЖЕНА ПАТОЛОГІЯ"

    return title, "\n".join(verdict_lines)

class ECGProcessor:
    def __init__(self, img_path):
        self.img_path = img_path
        self.original = cv2.imread(img_path)
        self.signal = None
        self.peaks = None
        self.cut_point = 0 
        self.pixels_per_sec = 0
        
    def extract_signal(self):
        green_channel = self.original[:, :, 1]
        _, binary = cv2.threshold(green_channel, 50, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            clean_mask = np.zeros_like(binary)
            clean_mask[labels == largest_label] = 255
        else:
            clean_mask = binary

        signal = []
        height, width = clean_mask.shape
        for x in range(width):
            col = clean_mask[:, x]
            pixels = np.where(col > 0)[0]
            if len(pixels) > 0:
                y_center = np.mean(pixels)
                val = height - y_center 
                signal.append(val)
            else:
                signal.append(signal[-1] if signal else height/2)
        
        self.signal = np.array(signal)
        self.cut_point = int(len(self.signal) * 0.15)
        median_val = np.median(self.signal)
        self.signal[:self.cut_point] = median_val
        return self.signal

    def detect_peaks(self, duration_sec):
        if duration_sec <= 0: duration_sec = 1
        self.pixels_per_sec = len(self.signal) / duration_sec
        live_signal = self.signal[self.cut_point:]
        if len(live_signal) == 0: live_signal = self.signal
        max_val = np.max(live_signal)
        min_height = max_val * 0.60 
        min_distance = self.pixels_per_sec * 0.25 
        peaks, _ = find_peaks(self.signal, height=min_height, distance=min_distance)
        peaks = peaks[peaks > self.cut_point]
        self.peaks = peaks
        return peaks

    def generate_plot(self, output_path):
        plt.figure(figsize=(10, 4))
        plt.style.use('dark_background')
        plt.plot(self.signal, color='#00ff00', linewidth=1, label="ЕКГ")
        if self.peaks is not None:
            plt.plot(self.peaks, self.signal[self.peaks], "rx", markersize=10, markeredgewidth=2)
        plt.axvline(x=self.cut_point, color='cyan', linestyle='--')
        plt.title(f"ECG Analysis v14.0")
        plt.grid(True, alpha=0.1)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def get_ai_prediction(signal, peaks):
    if not ai_available or len(peaks) == 0: return "N/A"
    crops = []
    for p in peaks:
        start, end = p - 60, p + 120
        if start < 0 or end >= len(signal): continue
        beat = signal[start:end]
        beat = resample(beat, 187)
        beat = (beat - beat.min()) / (beat.max() - beat.min() + 1e-6)
        crops.append(beat)
    if not crops: return "Error"
    x = np.array(crops).reshape(-1, 187, 1)
    preds = model.predict(x, verbose=0)
    votes = {v: 0 for v in classes_dict.values()}
    return max(votes, key=votes.get)

def calculate_metrics_v14(peaks, pixels_per_sec):
    if len(peaks) < 2: return 0, 0
    rr_pixels = np.diff(peaks)
    rr_sec = rr_pixels / pixels_per_sec
    median_rr = np.median(rr_sec)
    if median_rr > 0: bpm = int(60 / median_rr)
    else: bpm = 0
    mean_rr = np.mean(rr_sec)
    std_rr = np.std(rr_sec)
    cv = std_rr / mean_rr
    return bpm, cv

def generate_report_text(bpm, cv, ai_res, animal_type, weight, warnings):
    min_norm, max_norm = get_reference_values(animal_type, weight)
    main_title, details_text = get_full_diagnosis(bpm, min_norm, max_norm, cv, ai_res, animal_type, warnings)
    
    icon = "🐱" if "cat" in animal_type else "🐶"
    
    return (
        f"📋 **ВЕТЕРИНАРНИЙ ЗВІТ** ({icon} {weight} кг)\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"{main_title}\n\n"
        f"**Деталі аналізу:**\n"
        f"{details_text}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🔢 **Показники:**\n"
        f"• ЧСС: {bpm} уд/хв (Норма: {min_norm}-{max_norm})\n"
        f"• Варіабельність (CV): {int(cv*100)}%\n"
        f"• Морфологія (AI): {ai_res}"
    )

@dp.message(Command("start"))
async def start_handler(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer("👋 **Vet ECG v14.0 (Grand Diagnostic)**\nОберіть пацієнта:", reply_markup=get_animal_keyboard())

@dp.callback_query(F.data.startswith("animal_"))
async def animal_selected(callback: CallbackQuery, state: FSMContext):
    await state.update_data(animal_type=callback.data)
    await callback.message.edit_text(f"✅ Ви обрали {callback.data}. Введіть **вагу (кг)**:")
    await state.set_state(ECGState.waiting_for_weight)
    await callback.answer()

@dp.message(StateFilter(ECGState.waiting_for_weight))
async def weight_handler(message: types.Message, state: FSMContext):
    try: weight = float(message.text.replace(',', '.'))
    except: return
    await state.update_data(weight=weight)
    await message.answer(f"⚖️ {weight} кг. Метод аналізу:", reply_markup=get_method_keyboard())
    await state.set_state(ECGState.waiting_for_method)

@dp.callback_query(F.data == "method_photo")
async def method_photo_selected(callback: CallbackQuery, state: FSMContext):
    await state.update_data(method="photo")
    await callback.message.edit_text("📸 Надішліть **фото ЕКГ**.")
    await state.set_state(ECGState.waiting_for_photo)
    await callback.answer()

@dp.message(F.photo)
async def photo_handler(message: types.Message, state: FSMContext):
    data = await state.get_data()
    file = await bot.get_file(message.photo[-1].file_id)
    path = f"ecg_{message.from_user.id}.jpg"
    await bot.download_file(file.file_path, path)
    await state.update_data(img_path=path)
    await message.answer("⏱ Скільки секунд на записі?")
    await state.set_state(ECGState.waiting_for_duration)

@dp.message(StateFilter(ECGState.waiting_for_duration))
async def result_photo_handler(message: types.Message, state: FSMContext):
    try: duration = float(message.text.replace(',', '.'))
    except: return
    data = await state.get_data()
    
    msg = await message.answer("⏳ Глибокий аналіз...")

    try:
        proc = ECGProcessor(data['img_path'])
        signal = proc.extract_signal()
        peaks = proc.detect_peaks(duration)
        
        if len(peaks) < 2:
            await message.answer("❌ Замало даних.")
            return

        bpm, cv = calculate_metrics_v14(peaks, proc.pixels_per_sec)
        ai_res = get_ai_prediction(signal, peaks)
        
        warnings, suspicion_score = analyze_pathologies(signal, peaks, cv, ai_res)

        report = generate_report_text(bpm, cv, ai_res, data['animal_type'], data['weight'], warnings)
        
        plot_path = f"plot_{message.from_user.id}.png"
        proc.generate_plot(plot_path)
        
        await bot.send_photo(message.chat.id, FSInputFile(plot_path), caption=report, parse_mode="Markdown")
        os.remove(plot_path)
        if os.path.exists(data['img_path']): os.remove(data['img_path'])
        await message.answer("Новий пацієнт:", reply_markup=get_animal_keyboard())
        await state.clear()

    except Exception as e:
        await message.answer(f"Error: {e}")
        logging.error(e)

@dp.callback_query(F.data == "method_grid")
async def method_grid_selected(callback: CallbackQuery, state: FSMContext):
    await state.update_data(method="grid")
    await callback.message.edit_text("Швидкість?", reply_markup=get_speed_keyboard())
    await callback.answer()

@dp.callback_query(F.data.startswith("speed_"))
async def speed_selected(callback: CallbackQuery, state: FSMContext):
    speed = 25 if "25" in callback.data else 50
    await state.update_data(speed=speed)
    await callback.message.edit_text("Кількість **маленьких клітинок** між R-R:")
    await state.set_state(ECGState.waiting_for_squares)
    await callback.answer()

@dp.message(StateFilter(ECGState.waiting_for_squares))
async def result_grid_handler(message: types.Message, state: FSMContext):
    try: squares = float(message.text.replace(',', '.'))
    except: return
    data = await state.get_data()
    constant = 1500 if data['speed'] == 25 else 3000
    bpm = int(constant / squares)
    
    report = generate_report_text(bpm, 0, "N/A (Ручний ввід)", data['animal_type'], data['weight'], [])
    
    await message.answer(f"🧮 **ПО КЛІТИНКАХ**\n{report}", parse_mode="Markdown")
    await message.answer("Новий пацієнт:", reply_markup=get_animal_keyboard())
    await state.clear()

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())