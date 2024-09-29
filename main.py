# main.py

import argparse
from database_builder import build_database
from feature_extractor import FeatureExtractor
from database_manager import DatabaseManager
import numpy as np
import faiss
import os

def search_duplicate(video_path, threshold=0.5):
    extractor = FeatureExtractor()
    db_manager = DatabaseManager()
    features = extractor.extract(video_path)
    if features is None:
        print("Не удалось извлечь признаки из видео.")
        return

    db_features, video_ids = db_manager.get_all_features()
    if len(db_features) == 0:
        print("База данных пуста.")
        return

    # Создаем индекс Faiss
    d = features.shape[0]
    index = faiss.IndexFlatL2(d)
    index.add(db_features.astype('float32'))

    D, I = index.search(np.array([features]).astype('float32'), k=1)
    distance = D[0][0]
    closest_video_id = video_ids[I[0][0]]

    print(f"Минимальное расстояние: {distance}")
    if distance < threshold:
        print(f"Видео является дубликатом видео с ID: {closest_video_id}")
    else:
        print("Видео уникально и не является дубликатом.")

def main():
    parser = argparse.ArgumentParser(description='Поиск дубликатов видео.')
    parser.add_argument('--build-db', action='store_true', help='Построить базу данных из видеофайлов.')
    parser.add_argument('--videos-dir', type=str, default='videos', help='Директория с видеофайлами.')
    parser.add_argument('--search', type=str, help='Путь к видео для поиска дубликата.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Порог для определения дубликата.')
    parser.add_argument('--delete', type=str, help='ID видео для удаления из базы данных.')

    args = parser.parse_args()

    if args.build_db:
        build_database(args.videos_dir)
    elif args.search:
        search_duplicate(args.search, args.threshold)
    elif args.delete:
        db_manager = DatabaseManager()
        db_manager.delete_video(args.delete)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
