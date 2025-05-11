import mediapipe as mp
import numpy as np
from .base_model import FingerDetectorBaseModel
import cv2
from typing import Tuple, Union, Optional, List

class MediaPipeModel(FingerDetectorBaseModel):
    def __init__(self):
        super().__init__()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def is_palm_open(self, hand_landmarks) -> bool:
        """
        Проверяет, открыта ли ладонь, сравнивая y-координаты кончиков пальцев
        с их суставами. Ладонь считается открытой, если все пальцы выпрямлены вверх.
        """
        # Индексы кончиков пальцев и их суставов
        finger_tips = [8, 12, 16, 20]  # Кончики указательного, среднего, безымянного и мизинца
        finger_pips = [6, 10, 14, 18]  # Суставы этих пальцев
        
        # Считаем выпрямленные пальцы
        straight_fingers = 0
        
        # Проверяем каждый палец
        for tip, pip in zip(finger_tips, finger_pips):
            # Если кончик пальца ниже сустава, значит палец выпрямлен
            if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
                straight_fingers += 1
        
        # Ладонь открыта, если выпрямлены все пальцы
        return straight_fingers >= len(finger_tips)
        
    def get_finger(self, frame) -> Tuple[float, float]:
        """
        Определяет положение указательного пальца с помощью MediaPipe.
        """
        try:
            # Конвертируем BGR в RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Получаем результаты
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                # Берем первую обнаруженную руку
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Получаем координаты указательного пальца (точка 8)
                index_finger = hand_landmarks.landmark[8]
                
                # Конвертируем нормализованные координаты в пиксели
                h, w = frame.shape[:2]
                x_px = int(index_finger.x * w)
                y_px = int(index_finger.y * h)
                
                return float(x_px), float(y_px)
                
        except Exception as e:
            print(f"Error in MediaPipe model: {e}")
        
        return 0.0, 0.0
        
    def get_hand_landmarks(self, frame) -> Optional[List[Tuple[float, float]]]:
        """
        Возвращает все ключевые точки руки в виде списка координат.
        """
        try:
            # Конвертируем BGR в RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Получаем результаты
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                # Берем первую обнаруженную руку
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Конвертируем все точки в пиксели
                h, w = frame.shape[:2]
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    x_px = int(landmark.x * w)
                    y_px = int(landmark.y * h)
                    landmarks.append((x_px, y_px))
                
                return landmarks
                
        except Exception as e:
            print(f"Error getting hand landmarks: {e}")
        
        return None
        
    def cleanup(self):
        """Clean up resources used by the model"""
        if hasattr(self, 'hands'):
            self.hands.close() 