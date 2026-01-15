import asyncio
import logging
import os
import numpy as np
import cv2
import pickle
from concurrent.futures import ThreadPoolExecutor
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import FSInputFile, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from tensorflow.keras.models import load_model
from scipy.signal import find_peaks, resample
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

API_TOKEN = 'bottoken'  
MODEL_PATH = 'ecg_model.h5'
CLASSES_PATH = 'classes.pkl'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if not API_TOKEN or API_TOKEN == 'YOUR_TOKEN_HERE' or 'AIzaSy' in API_TOKEN:
    logging.error("❌ ПОМИЛКА: Невалідний токен Telegram!")
    logging.error("Отримайте токен від @BotFather в Telegram")
    logging.error("Формат токена: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz")
    raise ValueError("Невалідний токен Telegram. Отримайте токен від @BotFather")

try:
    bot = Bot(token=API_TOKEN)
    dp = Dispatcher()
except Exception as e:
    logging.error(f"❌ Помилка ініціалізації бота: {e}")
    raise

from aiogram import BaseMiddleware
from typing import Callable, Dict, Any, Awaitable

class LoggingMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[Any, Dict[str, Any]], Awaitable[Any]],
        event: Any,
        data: Dict[str, Any]
    ) -> Any:
        if hasattr(event, 'from_user') and event.from_user:
            logging.info(f"📨 Повідомлення від {event.from_user.id}: {getattr(event, 'text', 'без тексту')}")
        try:
            result = await handler(event, data)
            return result
        except Exception as e:
            logging.error(f"❌ Помилка в обробнику: {e}", exc_info=True)
            raise

dp.message.middleware(LoggingMiddleware())

executor = ThreadPoolExecutor(max_workers=2)

ai_available = False
model = None
classes_dict = None
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
        model = load_model(MODEL_PATH)
        with open(CLASSES_PATH, 'rb') as f:
            classes_dict = pickle.load(f)
        ai_available = True
        logging.info("✅ AI модель завантажена успішно")
    else:
        logging.warning("⚠️ AI модель не знайдено.")
except Exception as e:
    logging.error(f"Помилка AI: {e}")

class ECGState(StatesGroup):
    waiting_for_animal_type = State()
    waiting_for_weight = State()
    waiting_for_age = State()
    waiting_for_photo = State()
    waiting_for_duration = State()

_animal_keyboard = None

def get_animal_keyboard():
    """Створює клавіатуру вибору типу тварини"""
    global _animal_keyboard
    if _animal_keyboard is None:
        keyboard = ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="🐱 Кіт"), KeyboardButton(text="🐶 Собака")],
                [KeyboardButton(text="🧑 Людина")],
            ],
            resize_keyboard=True,
            one_time_keyboard=True
        )
        _animal_keyboard = keyboard
    return _animal_keyboard

def get_reference_values(animal_type, weight):
    min_norm, max_norm = 0, 0
    if "cat" in animal_type:
        min_norm, max_norm = 120, 200 
    elif "dog" in animal_type:
        if weight < 5: min_norm, max_norm = 100, 160
        elif weight < 15: min_norm, max_norm = 80, 140
        elif weight < 30: min_norm, max_norm = 70, 120
        elif weight < 50: min_norm, max_norm = 60, 100
        else: min_norm, max_norm = 50, 90
    elif "human" in animal_type or "patient" in animal_type:
        # Typical adult resting heart rate
        min_norm, max_norm = 60, 100
    return min_norm, max_norm

def analyze_pathologies(signal, peaks, cv, ai_verdict, pixels_per_sec):
    """Розширений аналіз патологій ЕКГ"""
    warnings = []
    suspicion_score = 0
    detailed_metrics = {}
    
    if len(peaks) < 3: 
        return warnings, 0, detailed_metrics

    amplitudes = signal[peaks]
    amplitude_cv = np.std(amplitudes) / (np.mean(amplitudes) + 1e-6)
    detailed_metrics['amplitude_cv'] = amplitude_cv
    
    if amplitude_cv > 0.15 and cv < 0.15:
        warnings.append("⚠️ **Електрична альтернація** (R-зубці різної висоти).\n  _👉 Виключіть випіт у перикард (тампонаду)._")
        suspicion_score += 2

    t_ratios = []
    t_waves = []
    if len(peaks) > 1:
        for i in range(len(peaks) - 1):
            start, end = peaks[i], peaks[i+1]
            margin = int((end - start) * 0.15)
            segment = signal[start+margin : end-margin]
            if len(segment) > 0:
                max_t = np.max(segment)
                max_r = signal[peaks[i]]
                if max_r > 1: 
                    t_ratios.append(max_t / max_r)
                    t_waves.append(max_t)
    
    if t_ratios:
        avg_t_ratio = np.mean(t_ratios)
        detailed_metrics['t_ratio'] = avg_t_ratio
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

    qrs_widths = []
    qrs_amplitudes = []
    for p in peaks:
        search_start = max(0, p - int(pixels_per_sec * 0.1))  # ~100ms до піку
        search_end = min(len(signal), p + int(pixels_per_sec * 0.1))  # ~100ms після піку
        
        segment = signal[search_start:search_end]
        if len(segment) > 0:
            baseline = np.median(signal[max(0, p-50):p])
            qrs_amplitude = signal[p] - baseline
            qrs_amplitudes.append(qrs_amplitude)
            
            threshold = baseline + (signal[p] - baseline) * 0.3
            above_threshold = np.where(segment > threshold)[0]
            if len(above_threshold) > 0:
                qrs_width = (above_threshold[-1] - above_threshold[0]) / pixels_per_sec * 1000  # в мс
                qrs_widths.append(qrs_width)
    
    if qrs_widths:
        avg_qrs_width = np.mean(qrs_widths)
        detailed_metrics['qrs_width_ms'] = avg_qrs_width
        if avg_qrs_width > 80:  
            warnings.append("📏 **Розширені QRS комплекси** (>80мс).\n  _👉 Можлива блокада або шлуночкова гіпертрофія._")
            suspicion_score += 1
    
    st_segments = []
    if len(peaks) > 1:
        for i in range(len(peaks) - 1):
            p = peaks[i]
            st_start = min(len(signal)-1, p + int(pixels_per_sec * 0.08))  
            st_end = min(len(signal)-1, p + int(pixels_per_sec * 0.16)) 
            
            if st_end > st_start:
                st_segment = signal[st_start:st_end]# R-зубець як референс
                st_deviation = np.mean(st_segment) - baseline
                st_segments.append(st_deviation)
    
    if st_segments:
        avg_st_dev = np.mean(st_segments)
        detailed_metrics['st_deviation'] = avg_st_dev
        st_dev_percent = (avg_st_dev / (np.mean(amplitudes) + 1e-6)) * 100
        if abs(st_dev_percent) > 20:  
            if st_dev_percent > 0:
                warnings.append("📈 **Підйом сегмента ST** (можлива ішемія/інфаркт).\n  _👉 Термінова консультація кардіолога!_")
            else:
                warnings.append("📉 **Депресія сегмента ST** (можлива ішемія).\n  _👉 Перевірте коронарний кровотік._")
            suspicion_score += 2

    if cv > 0.25 and len(rr_intervals) > 5:
        irregularity = np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-6)
        if irregularity > 0.3:
            warnings.append("💔 **Висока нерегулярність ритму** (можлива фібриляція передсердь).\n  _👉 Потрібна детальна діагностика._")
            suspicion_score += 1

    return warnings, suspicion_score, detailed_metrics

def get_full_diagnosis(bpm, min_norm, max_norm, cv, ai_verdict, animal_type, warnings, detailed_metrics=None):
    """Розширена діагностика з детальною інтерпретацією"""
    verdict_lines = []
    severity = "green"
    recommendations = []

    if bpm < min_norm:
        bpm_deviation = ((min_norm - bpm) / min_norm) * 100
        verdict_lines.append(f"• **Частота:** 🔴 **Брадикардія** ({bpm} < {min_norm} уд/хв)")
        if bpm_deviation > 30:
            verdict_lines.append(f"  └─ Виражена брадикардія (на {int(bpm_deviation)}% нижче норми)")
            recommendations.append("Перевірте електроліти, функцію щитоподібної залози")
            severity = "red"
        else:
            severity = "yellow"
            recommendations.append("Моніторинг частоти серцевих скорочень")
    elif bpm > max_norm:
        bpm_deviation = ((bpm - max_norm) / max_norm) * 100
        verdict_lines.append(f"• **Частота:** 🔴 **Тахікардія** ({bpm} > {max_norm} уд/хв)")
        if bpm_deviation > 20:
            verdict_lines.append(f"  └─ Виражена тахікардія (на {int(bpm_deviation)}% вище норми)")
            recommendations.append("Термінова оцінка: можливі причини - стрес, біль, гіпертиреоз, серцева недостатність")
            severity = "red"
        else:
            severity = "yellow"
            recommendations.append("Перевірте стан стресу, біль, температуру тіла")
    else:
        verdict_lines.append(f"• **Частота:** ✅ Нормосистолія ({bpm} уд/хв)")

    if cv > 0.15:
        if "dog" in animal_type and not warnings and ai_verdict in ["Normal (N)", "Норма (N)"] and cv < 0.35:
            verdict_lines.append(f"• **Ритм:** ⚠️ Синусова аритмія (CV {int(cv*100)}%)")
            verdict_lines.append("  └─ Фізіологічна норма для собак")
        else:
            cv_severity = "висока" if cv > 0.25 else "помірна"
            verdict_lines.append(f"• **Ритм:** ❌ **Нерегулярний** (CV {int(cv*100)}% - {cv_severity} варіабельність)")
            if severity == "green": 
                severity = "yellow"
            if cv > 0.25:
                recommendations.append("Детальна діагностика ритму - можлива фібриляція передсердь")
    else:
        verdict_lines.append(f"• **Ритм:** ✅ Правильний (CV {int(cv*100)}%)")

    if warnings:
        verdict_lines.append("\n**🔍 Виявлені відхилення:**")
        for w in warnings:
            verdict_lines.append(f"• {w}")
            if "ГІПЕР" in w or "Альтернація" in w or "ST" in w: 
                severity = "red"
            elif severity == "green": 
                severity = "yellow"

    if "V" in str(ai_verdict) or "VEB" in str(ai_verdict):
        verdict_lines.append("\n• **Морфологія:** 🔴 **Шлуночкові екстрасистоли (VPC)**")
        verdict_lines.append("  └─ Потрібна оцінка частоти та характеру екстрасистол")
        recommendations.append("Холтер-моніторинг для оцінки частоти VPC")
        severity = "red"
    elif "S" in str(ai_verdict):
        verdict_lines.append("\n• **Морфологія:** 🟡 Надшлуночкові екстрасистоли")
        verdict_lines.append("  └─ Зазвичай менш небезпечні, але потребують моніторингу")
        if severity == "green": 
            severity = "yellow"
    elif ai_verdict not in ["N/A", "Error"]:
        verdict_lines.append(f"\n• **Морфологія:** ✅ {ai_verdict}")

    if detailed_metrics:
        if 'qrs_width_ms' in detailed_metrics:
            qrs_w = detailed_metrics['qrs_width_ms']
            if qrs_w > 80:
                verdict_lines.append(f"\n• **QRS ширина:** ⚠️ {qrs_w:.1f} мс (норма: 40-70 мс)")
            else:
                verdict_lines.append(f"\n• **QRS ширина:** ✅ {qrs_w:.1f} мс")
        
        if 'st_deviation' in detailed_metrics:
            st_dev = detailed_metrics['st_deviation']
            if abs(st_dev) > 0.1:
                st_sign = "підйом" if st_dev > 0 else "депресія"
                verdict_lines.append(f"• **Сегмент ST:** ⚠️ {st_sign} {abs(st_dev):.2f}")

    title = ""
    if severity == "green": 
        title = "✅ КЛІНІЧНА НОРМА"
    elif severity == "yellow": 
        title = "⚠️ ПОМІРНІ ВІДХИЛЕННЯ"
    else: 
        title = "🚨 ВИРАЖЕНА ПАТОЛОГІЯ"

    if recommendations:
        verdict_lines.append(f"\n**💡 Рекомендації:**")
        for i, rec in enumerate(recommendations, 1):
            verdict_lines.append(f"{i}. {rec}")

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

        height, width = clean_mask.shape
        signal = np.zeros(width, dtype=np.float32)
        
        for x in range(width):
            col = clean_mask[:, x]
            pixels = np.where(col > 0)[0]
            if len(pixels) > 0:
                y_center = np.mean(pixels)
                signal[x] = height - y_center
            else:
                signal[x] = signal[x-1] if x > 0 else height/2
        
        self.signal = signal
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
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='black')
        ax.set_facecolor('black')
        ax.plot(self.signal, color='#00ff00', linewidth=1, label="ЕКГ")
        if self.peaks is not None and len(self.peaks) > 0:
            ax.plot(self.peaks, self.signal[self.peaks], "rx", markersize=10, markeredgewidth=2)
        ax.axvline(x=self.cut_point, color='cyan', linestyle='--')
        ax.set_title("ECG Analysis v14.0", color='white')
        ax.grid(True, alpha=0.1)
        ax.tick_params(colors='white')
        plt.tight_layout()
        plt.savefig(output_path, facecolor='black', dpi=100, bbox_inches='tight')
        plt.close(fig)

def get_ai_prediction(signal, peaks):
    if not ai_available or len(peaks) == 0 or model is None or classes_dict is None:
        return "N/A"
    
    crops = []
    for p in peaks:
        start, end = p - 60, p + 120
        if start < 0 or end >= len(signal): continue
        beat = signal[start:end]
        beat = resample(beat, 187)
        beat_min, beat_max = beat.min(), beat.max()
        if beat_max - beat_min > 1e-6:
            beat = (beat - beat_min) / (beat_max - beat_min)
        crops.append(beat)
    
    if not crops: 
        return "Error"
    
    x = np.array(crops, dtype=np.float32).reshape(-1, 187, 1)
    preds = model.predict(x, verbose=0, batch_size=min(32, len(crops)))
    
    votes = {v: 0 for v in classes_dict.values()}
    class_indices = list(classes_dict.keys())
    
    for pred in preds:
        predicted_class_idx = np.argmax(pred)
        if predicted_class_idx < len(class_indices):
            predicted_class = classes_dict[class_indices[predicted_class_idx]]
            votes[predicted_class] = votes.get(predicted_class, 0) + 1
    
    if not votes or max(votes.values()) == 0:
        return "N/A"
    
    return max(votes, key=votes.get)

def calculate_metrics_v14(peaks, pixels_per_sec, signal=None):
    """Розширений розрахунок метрик ЕКГ"""
    if len(peaks) < 2: 
        return {
            'bpm': 0, 
            'cv': 0, 
            'rr_intervals': [],
            'mean_rr': 0,
            'min_rr': 0,
            'max_rr': 0
        }
    
    rr_pixels = np.diff(peaks)
    rr_sec = rr_pixels / pixels_per_sec
    median_rr = np.median(rr_sec)
    
    if median_rr > 0: 
        bpm = int(60 / median_rr)
    else: 
        bpm = 0
    
    mean_rr = np.mean(rr_sec)
    std_rr = np.std(rr_sec)
    cv = std_rr / (mean_rr + 1e-6)
    
    return {
        'bpm': bpm,
        'cv': cv,
        'rr_intervals': rr_sec,
        'mean_rr': mean_rr,
        'min_rr': np.min(rr_sec) if len(rr_sec) > 0 else 0,
        'max_rr': np.max(rr_sec) if len(rr_sec) > 0 else 0,
        'std_rr': std_rr,
        'median_rr': median_rr
    }

def generate_report_text(metrics, ai_res, animal_type, weight, warnings, detailed_metrics=None):
    """Генерація детального звіту з розшифровкою"""
    min_norm, max_norm = get_reference_values(animal_type, weight)
    bpm = metrics.get('bpm', 0)
    cv = metrics.get('cv', 0)
    
    main_title, details_text = get_full_diagnosis(
        bpm, min_norm, max_norm, cv, ai_res, animal_type, warnings, detailed_metrics
    )
    
    if 'cat' in animal_type:
        icon = "🐱"
        subject_label = f"{weight} кг"
    elif 'dog' in animal_type:
        icon = "🐶"
        subject_label = f"{weight} кг"
    else:
        icon = "🧑"
        subject_label = f"{weight} років"
    
    metrics_text = f"🔢 **Детальні показники:**\n"
    metrics_text += f"• ЧСС: {bpm} уд/хв (Норма: {min_norm}-{max_norm})\n"
    metrics_text += f"• Варіабельність (CV): {int(cv*100)}%\n"
    
    if 'mean_rr' in metrics and metrics['mean_rr'] > 0:
        metrics_text += f"• Середній RR: {metrics['mean_rr']*1000:.0f} мс\n"
    if 'min_rr' in metrics and 'max_rr' in metrics:
        metrics_text += f"• RR діапазон: {metrics['min_rr']*1000:.0f}-{metrics['max_rr']*1000:.0f} мс\n"
    
    metrics_text += f"• Морфологія (AI): {ai_res}\n"
    
    if detailed_metrics:
        if 'qrs_width_ms' in detailed_metrics:
            metrics_text += f"• Ширина QRS: {detailed_metrics['qrs_width_ms']:.1f} мс\n"
        if 't_ratio' in detailed_metrics:
            metrics_text += f"• T/R співвідношення: {detailed_metrics['t_ratio']:.2f}\n"
    
    header_name = "ВЕТЕРИНАРНИЙ ЗВІТ ЕКГ" if 'animal' in animal_type else "ЗВІТ ЕКГ"
    return (
        f"📋 **{header_name}** ({icon} {subject_label})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**{main_title}**\n\n"
        f"{details_text}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{metrics_text}"
    )

@dp.message(Command("start"))
async def start_handler(message: types.Message, state: FSMContext):
    try:
        logging.info(f"📥 Отримано команду /start від користувача {message.from_user.id}")
        await state.clear()
        
        welcome_text = (
            "👋 **Вітаю в ECG Bot!**\n\n"
            "🔬 Я допоможу вам розшифрувати електрокардіограму тварини або людини.\n\n"
            "📸 Просто надішліть фото ЕКГ стрічки, і я проведу повний аналіз:\n"
            "• Розрахунок частоти серцевих скорочень\n"
            "• Аналіз ритму та регулярності\n"
            "• Виявлення патологій\n"
            "• Детальна інтерпретація результатів\n\n"
            "👥 **Оберіть тип пацієнта:**"
        )
        
        await message.answer(
            welcome_text,
            reply_markup=get_animal_keyboard(),
            parse_mode="Markdown"
        )
        await state.set_state(ECGState.waiting_for_animal_type)
        logging.info(f"✅ Відправлено відповідь користувачу {message.from_user.id}")
    except Exception as e:
        logging.error(f"❌ Помилка в start_handler: {e}", exc_info=True)
        try:
            await message.answer(
                "👋 Вітаю в Vet ECG Bot!\n\nОберіть тип пацієнта:",
                reply_markup=get_animal_keyboard()
            )
            await state.set_state(ECGState.waiting_for_animal_type)
        except Exception as e2:
            logging.error(f"❌ Критична помилка відправки повідомлення: {e2}", exc_info=True)

@dp.message(StateFilter(ECGState.waiting_for_animal_type))
async def animal_selected(message: types.Message, state: FSMContext):
    text = message.text.lower()
    animal_type = None
    
    if "кіт" in text or "cat" in text or "🐱" in text:
        animal_type = "animal_cat"
        animal_name = "Кіт 🐱"
    elif "собака" in text or "dog" in text or "🐶" in text:
        animal_type = "animal_dog"
        animal_name = "Собака 🐶"
    elif "людина" in text or "human" in text or "🧑" in text:
        animal_type = "patient_human"
        animal_name = "Людина 🧑"
    else:
        await message.answer(
            "❌ Будь ласка, оберіть тип тварини з клавіатури або напишіть 'Кіт' або 'Собака'.",
            reply_markup=get_animal_keyboard()
        )
        return
    
    await state.update_data(animal_type=animal_type)
    # For humans ask for age, for animals ask for weight
    if animal_type == 'patient_human':
        await message.answer(
            f"✅ Ви обрали: **{animal_name}**\n\n"
            f"🧾 **Введіть вік пацієнта в роках:**\n"
            f"_(Наприклад: 34 або 71)_",
            reply_markup=ReplyKeyboardRemove(),
            parse_mode="Markdown"
        )
        await state.set_state(ECGState.waiting_for_age)
    else:
        await message.answer(
            f"✅ Ви обрали: **{animal_name}**\n\n"
            f"⚖️ **Введіть вагу тварини в кілограмах:**\n"
            f"_(Наприклад: 5.5 або 12)_",
            reply_markup=ReplyKeyboardRemove(),
            parse_mode="Markdown"
        )
        await state.set_state(ECGState.waiting_for_weight)

@dp.message(StateFilter(ECGState.waiting_for_weight))
async def weight_handler(message: types.Message, state: FSMContext):
    try: 
        weight = float(message.text.replace(',', '.'))
        if weight <= 0 or weight > 200:
            await message.answer("❌ Будь ласка, введіть коректну вагу (від 0.1 до 200 кг).")
            return
    except: 
        await message.answer("❌ Будь ласка, введіть число (наприклад: 5.5 або 12)")
        return
    
    data = await state.get_data()
    animal_name = "Кіт 🐱" if "cat" in data.get('animal_type', '') else "Собака 🐶"
    
    await state.update_data(weight=weight)
    await message.answer(
        f"✅ **Пацієнт:** {animal_name}\n"
        f"⚖️ **Вага:** {weight} кг\n\n"
        f"📸 **Тепер надішліть фото ЕКГ стрічки:**\n\n"
        f"💡 _Поради для кращого результату:_\n"
        f"• Фото має бути чітким та добре освітленим\n"
        f"• ЕКГ стрічка має бути видною на весь кадр\n"
        f"• Уникайте відблисків та тіней",
        parse_mode="Markdown"
    )
    await state.set_state(ECGState.waiting_for_photo)


@dp.message(StateFilter(ECGState.waiting_for_age))
async def age_handler(message: types.Message, state: FSMContext):
    try:
        age = int(float(message.text.replace(',', '.')))
        if age < 0 or age > 120:
            await message.answer("❌ Будь ласка, введіть коректний вік (0-120 років).")
            return
    except:
        await message.answer("❌ Будь ласка, введіть число (наприклад: 34 або 71)")
        return

    data = await state.get_data()
    await state.update_data(age=age)
    patient_name = "Людина 🧑"
    await message.answer(
        f"✅ **Пацієнт:** {patient_name}\n"
        f"🧾 **Вік:** {age} років\n\n"
        f"📸 **Тепер надішліть фото ЕКГ стрічки:**\n\n"
        f"💡 _Поради для кращого результату:_\n"
        f"• Фото має бути чітким та добре освітленим\n"
        f"• ЕКГ стрічка має бути видною на весь кадр\n"
        f"• Уникайте відблисків та тіней",
        parse_mode="Markdown"
    )
    await state.set_state(ECGState.waiting_for_photo)

@dp.message(F.photo, StateFilter(ECGState.waiting_for_photo))
async def photo_handler(message: types.Message, state: FSMContext):
    try:
        logging.info(f"📷 Отримано фото від користувача {message.from_user.id}")
        
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        path = f"ecg_{message.from_user.id}_{photo.file_unique_id}.jpg"
        
        await bot.download_file(file.file_path, path)
        await state.update_data(img_path=path)
        
        await message.answer(
            "✅ **Фото отримано!**\n\n"
            "⏱ **Скільки секунд запису міститься на фото?**\n\n"
            "💡 _Введіть тривалість запису в секундах (наприклад: 3.5 або 5)_",
            parse_mode="Markdown"
        )
        await state.set_state(ECGState.waiting_for_duration)
        logging.info(f"✅ Фото завантажено: {path}")
    except Exception as e:
        await message.answer(
            f"❌ **Помилка завантаження фото**\n\n"
            f"Спробуйте надіслати фото ще раз або перевірте підключення до інтернету."
        )
        logging.error(f"Помилка завантаження фото: {e}", exc_info=True)

def process_ecg_sync(img_path, duration, animal_type, weight):
    """Синхронна обробка ЕКГ для виконання в thread pool"""
    try:
        proc = ECGProcessor(img_path)
        signal = proc.extract_signal()
        peaks = proc.detect_peaks(duration)
        
        if len(peaks) < 2:
            return None, "❌ Замало даних."
        
        metrics = calculate_metrics_v14(peaks, proc.pixels_per_sec, signal)
        bpm = metrics.get('bpm', 0)
        cv = metrics.get('cv', 0)
        
        ai_res = get_ai_prediction(signal, peaks)
        warnings, suspicion_score, detailed_metrics = analyze_pathologies(
            signal, peaks, cv, ai_res, proc.pixels_per_sec
        )
        report = generate_report_text(metrics, ai_res, animal_type, weight, warnings, detailed_metrics)
        
        plot_path = f"plot_{os.getpid()}_{id(proc)}.png"
        proc.generate_plot(plot_path)
        
        del proc
        del signal
        
        return plot_path, report
    except Exception as e:
        logging.error(f"Помилка обробки ЕКГ: {e}", exc_info=True)
        return None, f"Помилка обробки: {e}"

@dp.message(StateFilter(ECGState.waiting_for_duration))
async def result_photo_handler(message: types.Message, state: FSMContext):
    try: 
        duration = float(message.text.replace(',', '.'))
        if duration <= 0 or duration > 60:
            await message.answer(
                "❌ **Невірна тривалість!**\n\n"
                "Будь ласка, введіть число від 0.1 до 60 секунд."
            )
            return
    except: 
        await message.answer(
            "❌ **Невірний формат!**\n\n"
            "Будь ласка, введіть число (наприклад: 3.5 або 5)"
        )
        return
    
    data = await state.get_data()
    img_path = data.get('img_path')
    
    if not img_path or not os.path.exists(img_path):
        await message.answer("❌ Фото не знайдено. Спробуйте знову.")
        await state.clear()
        return
    
    msg = await message.answer(
        "⏳ **Почато аналіз ЕКГ...**\n\n"
        "🔬 Обробка зображення та виявлення сигналу\n"
        "📊 Розрахунок метрик та аналіз патологій\n"
        "🤖 AI класифікація морфології\n\n"
        "_Це може зайняти кілька секунд..._",
        parse_mode="Markdown"
    )

    try:
        loop = asyncio.get_event_loop()
        patient_value = None
        if data.get('animal_type') and 'patient_human' in data.get('animal_type'):
            patient_value = data.get('age')
        else:
            patient_value = data.get('weight')

        plot_path, report = await loop.run_in_executor(
            executor,
            process_ecg_sync,
            img_path,
            duration,
            data['animal_type'],
            patient_value
        )
        
        if plot_path is None:
            await msg.edit_text(
                f"❌ **Помилка аналізу**\n\n{report}\n\n"
                f"Спробуйте надіслати інше фото або перевірте якість зображення.",
                parse_mode="Markdown"
            )
            # Cleanup
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except:
                    pass
            await state.clear()
            return

        await bot.send_photo(
            message.chat.id, 
            FSInputFile(plot_path), 
            caption=report, 
            parse_mode="Markdown"
        )
        
        try:
            os.remove(plot_path)
        except:
            pass
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
        except:
            pass
        
        await message.answer(
            "✅ **Аналіз завершено!**\n\n"
            "🔄 Для нового аналізу надішліть команду /start",
            reply_markup=ReplyKeyboardRemove(),
            parse_mode="Markdown"
        )
        await state.clear()

    except Exception as e:
        await msg.edit_text(
            f"❌ **Помилка під час аналізу**\n\n"
            f"Деталі: {str(e)}\n\n"
            f"Спробуйте надіслати інше фото або зверніться до підтримки.",
            parse_mode="Markdown"
        )
        logging.error(f"Помилка в result_photo_handler: {e}", exc_info=True)
        
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
        except:
            pass
        await state.clear()


async def main():
    global model 
    try:
        logging.info("🚀 Запуск бота...")
        
        try:
            bot_info = await bot.get_me()
            logging.info(f"✅ Бот підключено: @{bot_info.username} ({bot_info.first_name}, ID: {bot_info.id})")
        except Exception as e:
            error_msg = str(e)
            logging.error(f"❌ Помилка підключення до Telegram API: {error_msg}")
            if "Unauthorized" in error_msg or "401" in error_msg:
                logging.error("=" * 60)
                logging.error("⚠️ ТОКЕН НЕВАЛІДНИЙ АБО ВІДКЛИКАНИЙ!")
                logging.error("=" * 60)
                logging.error("Перевірте токен у файлі bot.py (рядок 19)")
                logging.error(f"Поточний токен: {API_TOKEN[:10]}...{API_TOKEN[-5:]}")
                logging.error("")
                logging.error("Щоб отримати новий токен:")
                logging.error("1. Відкрийте Telegram")
                logging.error("2. Знайдіть @BotFather")
                logging.error("3. Надішліть /newbot або /token")
                logging.error("4. Скопіюйте токен у форматі: 123456789:ABCdef...")
                logging.error("=" * 60)
            raise
        
        logging.info("📡 Початок polling...")
        await dp.start_polling(
            bot, 
            drop_pending_updates=True, 
            allowed_updates=["message"],
            handle_as_tasks=True
        )
    except Exception as e:
        logging.error(f"❌ Критична помилка: {e}", exc_info=True)
        if "Conflict" in str(e) or "getUpdates" in str(e):
            logging.error("⚠️ ВИЯВЛЕНО КОНФЛІКТ!")
            logging.error("Зупиніть всі інші екземпляри бота перед запуском.")
            logging.error("Або зачекайте 1-2 хвилини і спробуйте знову.")
    finally:
        logging.info("🛑 Зупинка бота...")
        executor.shutdown(wait=True)
        if 'model' in globals() and model is not None:
            try:
                del model
            except:
                pass

if __name__ == "__main__":
    try:
        import psutil  
        current_process = psutil.Process()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    if proc.info['cmdline'] and any('bot.py' in str(cmd) for cmd in proc.info['cmdline']):
                        if proc.info['pid'] != current_process.pid:
                            logging.warning(f"⚠️ Знайдено інший запущений процес бота (PID: {proc.info['pid']})")
                            logging.warning("Зупиніть його перед запуском нового екземпляра!")
            except (psutil.NoSuchProcess, psutil.AccessDenied):  
                pass
    except ImportError:
        pass
    except Exception as e:
        logging.debug(f"Помилка перевірки процесів: {e}")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("✅ Бот зупинено користувачем")
    except Exception as e:
        logging.error(f"❌ Несподівана помилка: {e}", exc_info=True)
    finally:
        executor.shutdown(wait=True)