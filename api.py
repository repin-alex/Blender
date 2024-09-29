import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Решение проблемы с дублирующимся OpenMP

import csv
import uuid
import numpy as np
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import yt_dlp
import shutil
from feature_extractor import FeatureExtractor
import faiss
import webbrowser
import uvicorn

app = FastAPI()

# Настройка шаблонов
templates = Jinja2Templates(directory="templates")

# Пути к CSV файлам
LINKS_CSV = "links.csv"
METRICS_CSV = "metrics.csv"

# Инициализация FeatureExtractor
extractor = FeatureExtractor()


# Функция для создания CSV файлов, если они не существуют
def create_csv_if_not_exists():
    if not os.path.exists(LINKS_CSV):
        with open(LINKS_CSV, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["created", "uuid", "link"])

    if not os.path.exists(METRICS_CSV):
        with open(METRICS_CSV, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["uuid", "vector"])


# Функция для загрузки данных из links.csv
def load_links():
    links = []
    if os.path.exists(LINKS_CSV):
        with open(LINKS_CSV, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                links.append(row)
    return links


# Функция для загрузки данных из metrics.csv
def load_metrics():
    metrics = []
    if os.path.exists(METRICS_CSV):
        with open(METRICS_CSV, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                vector = np.array([float(v) for v in row["vector"].split(",")])
                metrics.append({
                    "uuid": row["uuid"],
                    "vector": vector
                })
    return metrics


# Функция для сохранения ссылки в links.csv
def save_link_to_csv(video_uuid, link):
    with open(LINKS_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Дата и время
            video_uuid,  # Уникальный идентификатор
            link  # Ссылка на видео
        ])


# Функция для сохранения метрики в metrics.csv
def save_metrics_to_csv(video_uuid, vector):
    with open(METRICS_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            video_uuid,  # Уникальный идентификатор
            ",".join(map(str, vector))  # Преобразуем вектор в строку
        ])


# Функция для поиска ссылки в базе данных
def find_link_in_csv(link):
    links = load_links()
    for item in links:
        if item["link"] == link.strip():
            return item  # Ссылка найдена
    return None  # Ссылка не найдена


# Функция для загрузки видео с YouTube или другого ресурса
def download_video(video_url):
    ydl_opts = {'outtmpl': 'temp/%(id)s.%(ext)s', 'format': 'best'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)
        return os.path.join("temp", f"{info_dict['id']}.{info_dict['ext']}"), info_dict['id']


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_video(request: Request, video_url: str = Form(None), frame_step: int = Form(5)):
    try:
        temp_video_path = None
        video_uuid = None

        # Создаем CSV файлы, если их нет
        create_csv_if_not_exists()

        # Проверка ссылки в базе данных
        video_in_db = find_link_in_csv(video_url)
        if video_in_db:
            # Если видео уже есть в базе, возвращаем информацию о дубликате
            result = {
                "message": f"Видео уже в базе данных с ID: {video_in_db['uuid']}",
                "is_duplicate": True,
                "link": video_in_db['link']
            }
            return templates.TemplateResponse("index.html", {"request": request, "result": result})

        # Если ссылка отсутствует, загружаем видео
        try:
            temp_video_path, video_uuid = download_video(video_url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при загрузке видео по ссылке: {e}")

        # Извлечение признаков видео
        try:
            features = extractor.extract(temp_video_path, frame_step=frame_step)
            if features is None:
                os.remove(temp_video_path)
                raise HTTPException(status_code=400, detail="Не удалось извлечь признаки из видео.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при извлечении признаков из видео: {e}")

        # Сравнение метрик с базой данных
        metrics = load_metrics()
        if not metrics:
            is_duplicate = False
        else:
            db_vectors = np.array([item['vector'] for item in metrics])
            d = features.shape[0]
            index = faiss.IndexFlatL2(d)
            index.add(db_vectors.astype('float32'))
            D, I = index.search(np.array([features]).astype('float32'), k=1)
            distance = D[0][0]
            threshold = 0.5
            is_duplicate = distance < threshold

        # Если видео уникально, сохраняем ссылку и метрики
        if not is_duplicate:
            save_link_to_csv(video_uuid, video_url)
            save_metrics_to_csv(video_uuid, features)
            message = "Видео уникально и добавлено в базу данных."
        else:
            message = "Видео является дубликатом."

        result = {
            "message": message,
            "is_duplicate": is_duplicate,
            "link": video_url
        }

        return templates.TemplateResponse("index.html", {"request": request, "result": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сервера: {e}")


# Запуск приложения с автоматическим открытием браузера
if __name__ == "__main__":
    import threading
    threading.Timer(1.5, lambda: webbrowser.open_new("http://127.0.0.1:8000")).start()
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
