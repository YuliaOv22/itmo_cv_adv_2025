import cv2
import os
from ultralytics import YOLO
import numpy as np
from .base_model import FingerDetectorBaseModel
from typing import Tuple

class BaseYOLOModel(FingerDetectorBaseModel):
    def __init__(self, model_path: str):
        # Получаем абсолютный путь к файлу модели
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(current_dir, 'ml_models', 'best_big.pt')
        
        # Загружаем модель YOLO
        self.model = YOLO(model_path)
        
    def get_finger(self, frame) -> Tuple[float, float]:
        """
        Определяет положение указательного пальца с помощью YOLO.
        """
        try:
            # Получаем предсказания
            predictions = self.model.predict(source=frame, verbose=False)
            
            # Проверяем, есть ли предсказания
            if len(predictions) > 0 and len(predictions[0].keypoints) > 0:
                # Получаем ключевые точки
                kps = predictions[0].keypoints.xyn[0]
                
                # Получаем координаты указательного пальца (точка 13)
                x_norm, y_norm = kps[13].tolist()
                
                # Конвертируем нормализованные координаты в пиксели
                h, w = frame.shape[:2]
                x_px = int(x_norm * w)
                y_px = int(y_norm * h)
                
                return float(x_px), float(y_px)
                
        except Exception as e:
            print(f"Error in YOLO model: {e}")
        
        return 0.0, 0.0
        
    def cleanup(self):
        """Clean up resources used by the model"""
        if hasattr(self, 'model'):
            del self.model

class YOLOSmallModel(BaseYOLOModel):
    def __init__(self):
        super().__init__('best_small.pt')

class YOLOBigModel(BaseYOLOModel):
    def __init__(self):
        super().__init__('best_big.pt') 