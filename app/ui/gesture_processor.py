from PyQt5.QtCore import QThread, pyqtSignal, QMutex
from queue import Queue
from models.mediapipe_model import MediaPipeModel

class GestureProcessor(QThread):
    gesture_detected = pyqtSignal(bool)  # Сигнал с информацией о жесте стирания
    
    def __init__(self):
        super().__init__()
        self.model = MediaPipeModel()
        self.frame_queue = Queue(maxsize=1)
        self.running = True
        self.mutex = QMutex()
        self.processing = False
        
    def run(self):
        while self.running:
            if not self.frame_queue.empty() and not self.processing:
                self.processing = True
                frame = self.frame_queue.get()
                try:
                    # Получаем все ключевые точки руки
                    landmarks = self.model.get_hand_landmarks(frame)
                    if landmarks is not None:
                        # Проверяем, является ли жест кулаком
                        is_fist = self.check_fist_gesture(landmarks)
                        self.gesture_detected.emit(is_fist)
                except Exception as e:
                    print(f"Error processing gesture: {e}")
                finally:
                    self.processing = False
                    
    def check_fist_gesture(self, landmarks):
        """Проверяет, является ли жест кулаком"""
        # Получаем координаты кончиков пальцев и суставов
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Получаем координаты суставов пальцев
        index_pip = landmarks[6]
        middle_pip = landmarks[10]
        ring_pip = landmarks[14]
        pinky_pip = landmarks[18]
        
        # Проверяем, что все пальцы согнуты (кончики пальцев находятся ближе к ладони, чем суставы)
        fingers_bent = (
            index_tip[1] > index_pip[1] and  # Указательный палец
            middle_tip[1] > middle_pip[1] and  # Средний палец
            ring_tip[1] > ring_pip[1] and  # Безымянный палец
            pinky_tip[1] > pinky_pip[1]  # Мизинец
        )
        
        return fingers_bent
                    
    def add_frame(self, frame):
        if self.frame_queue.empty() and not self.processing:
            self.frame_queue.put(frame)
            
    def stop(self):
        self.running = False
        self.wait()
        if self.model:
            self.model.cleanup() 