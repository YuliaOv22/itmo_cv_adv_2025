from abc import ABC, abstractmethod
from typing import Tuple

class FingerDetectorBaseModel(ABC):
    @abstractmethod
    def get_finger(self, frame) -> Tuple[float, float]:
        """
        Метод для определения координат указательного пальца на кадре
        
        Args:
            frame: Кадр с камеры
            
        Returns:
            Tuple[float, float]: Координаты пальца (x, y) или (0.0, 0.0) если палец не найден
        """
        pass 