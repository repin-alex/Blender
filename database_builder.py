# database_builder.py

import os
from feature_extractor import FeatureExtractor
from database_manager import DatabaseManager

def build_database(videos_dir='videos'):
    extractor = FeatureExtractor()
    db_manager = DatabaseManager()
    video_files = [f for f in os.listdir(videos_dir) if os.path.isfile(os.path.join(videos_dir, f))]
    if not video_files:
        print(f"Нет видеофайлов в директории {videos_dir}")
        return

    # Очищаем текущую базу данных перед пересборкой
    db_manager.index.clear()
    db_manager.save_index()

    for video_file in video_files:
        video_path = os.path.join(videos_dir, video_file)
        print(f"Обработка {video_file}")
        features = extractor.extract(video_path)
        if features is not None:
            video_id = os.path.splitext(video_file)[0]
            db_manager.add_video(video_id, features)
        else:
            print(f"Признаки не извлечены для {video_file}")
    print("База данных пересобрана.")
