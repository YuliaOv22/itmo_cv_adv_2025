from ultralytics import YOLO
import cv2
import os
from .base_model import FingerDetectorBaseModel
from typing import Tuple

class YOLOSmallModel(FingerDetectorBaseModel):
    def __init__(self):
        super().__init__()
        # Получаем абсолютный путь к файлу модели
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(current_dir, 'ml_models', 'best_small.pt')
        
        try:
            # Загружаем модель YOLO small
            self.model = YOLO(model_path)
            print(f"YOLO Small model loaded from: {model_path}")
        except Exception as e:
            print(f"Error loading YOLO Small model: {e}")
            self.model = None
        
    def get_finger(self, frame) -> Tuple[float, float]:
        """
        Определяет положение указательного пальца с помощью YOLO small модели.
        """
        if self.model is None:
            print("YOLO Small model not initialized")
            return 0.0, 0.0
            
        try:
            # Получаем предсказания
            results = self.model.predict(frame, verbose=False)
            
            # Проверяем, есть ли предсказания
            if len(results) > 0 and hasattr(results[0], 'keypoints') and len(results[0].keypoints) > 0:
                # Получаем ключевые точки
                kps = results[0].keypoints.xyn[0]
                
                # Получаем координаты указательного пальца (точка 8)
                x_norm, y_norm = kps[8].tolist()
                
                # Конвертируем нормализованные координаты в пиксели
                h, w = frame.shape[:2]
                x_px = int(x_norm * w)
                y_px = int(y_norm * h)
                
                return float(x_px), float(y_px)
                
        except Exception as e:
            print(f"Error in YOLO small model: {e}")
        
        return 0.0, 0.0
        
    def cleanup(self):
        """Clean up resources used by the model"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model 
            self.model = None 