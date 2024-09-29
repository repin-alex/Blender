import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Решение проблемы с дублирующимся OpenMP

import shutil
import csv
import uuid
import numpy as np
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import faiss
from feature_extractor import FeatureExtractor
import yt_dlp
import uvicorn
import webbrowser
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Разрешение CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = FeatureExtractor()
templates = Jinja2Templates(directory="templates")

CSV_FILE = 'train.csv'

# Функция для создания файла train.csv, если он не существует
def create_csv_if_not_exists():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["created", "uuid", "link", "vector"])  # Обновленная структура файла

# Функция для сохранения информации о видео в train.csv
def save_video_data_to_csv(video_uuid, link, vector):
    create_csv_if_not_exists()
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Дата и время
            video_uuid,  # UUID видео
            link,  # Ссылка на видео
            vector  # Среднее значение метрики
        ])

# Функция для загрузки данных из train.csv
def load_video_data_from_csv():
    video_data = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                video_data.append({
                    "created": row["created"],
                    "uuid": row["uuid"],
                    "link": row["link"],
                    "vector": float(row["vector"])  # Среднее значение метрики
                })
    return video_data

# Проверка, есть ли ссылка в базе данных
def find_video_by_link(link):
    video_data = load_video_data_from_csv()
    for item in video_data:
        if item['link'] == link:
            return item  # Возвращаем данные о видео, если ссылка найдена
    return None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_video(request: Request, video_url: str = Form(None), file: UploadFile = File(None), frame_step: int = Form(5)):
    try:
        temp_video_path = None

        # Проверка ссылки в базе данных
        if video_url:
            video_in_db = find_video_by_link(video_url)
            if video_in_db:
                # Если видео уже есть в базе, возвращаем информацию о дубликате или уникальности
                result = {
                    "message": "Видео уже в базе данных.",
                    "uuid": video_in_db['uuid'],
                    "link": video_in_db['link']
                }
                return templates.TemplateResponse("index.html", {"request": request, "result": result})

            # Если видео нет в базе, загружаем его
            try:
                ydl_opts = {'outtmpl': 'temp/%(id)s.%(ext)s', 'format': 'best'}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(video_url, download=True)
                    temp_video_path = os.path.join("temp", f"{info_dict['id']}.{info_dict['ext']}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Ошибка при загрузке видео по ссылке: {e}")

        elif file:
            # Загружаем видеофайл
            try:
                if not os.path.exists("temp"):
                    os.makedirs("temp")
                temp_video_path = os.path.join("temp", file.filename)
                with open(temp_video_path, "wb") as buffer:
                    buffer.write(await file.read())
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Ошибка при сохранении загруженного файла: {e}")
        else:
            raise HTTPException(status_code=400, detail="Необходимо передать видео через URL или загрузить файл.")

        # Извлекаем признаки видео
        try:
            features = extractor.extract(temp_video_path, frame_step=frame_step)
            if features is None:
                os.remove(temp_video_path)
                raise HTTPException(status_code=400, detail="Не удалось извлечь признаки из видео.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при извлечении признаков из видео: {e}")

        # Вычисляем среднее значение метрики
        vector_mean = np.mean(features)

        # Загружаем все данные о видео из CSV для сравнения метрик
        video_data = load_video_data_from_csv()
        if not video_data:
            # Если база данных пуста, видео уникально
            is_duplicate = False
            closest_video_id = None
            distance = None
        else:
            # Сравниваем с другими видео по метрикам (сравниваем среднее значение метрик)
            db_vectors = np.array([item['vector'] for item in video_data])
            video_ids = [item['uuid'] for item in video_data]
            distances = np.abs(db_vectors - vector_mean)  # Расстояние между метриками
            closest_index = np.argmin(distances)
            distance = distances[closest_index]
            closest_video_id = video_ids[closest_index]
            threshold = 0.1  # Задаем порог для определения дубликата
            is_duplicate = distance < threshold

        # Если это дубликат, сохраняем информацию в CSV
        if is_duplicate:
            message = f"Видео является дубликатом видео с ID: {closest_video_id}"
            os.remove(temp_video_path)  # Удаляем временный файл
            result = {
                "message": "Это видео является дубликатом.",
                "uuid": closest_video_id,  # UUID оригинального видео
                "link": find_video_by_link(closest_video_id)['link']  # Ссылка на оригинальное видео
            }
        else:
            # Если видео уникально, добавляем его в базу данных
            message = "Видео уникально и не является дубликатом. Видео добавлено в базу данных."
            video_id = os.path.splitext(file.filename)[0] if file else info_dict['id']
            save_video_data_to_csv(video_id, video_url if video_url else file.filename, vector_mean)
            result = {
                "message": message,
                "uuid": video_id,
                "link": video_url if video_url else file.filename
            }

        return templates.TemplateResponse("index.html", {"request": request, "result": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера.")
    

# Запуск приложения с автоматическим открытием браузера
if __name__ == "__main__":
    import threading
    threading.Timer(1.5, lambda: webbrowser.open_new("http://127.0.0.1:8000")).start()
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
