from PyQt5.QtCore import QThread, pyqtSignal, QMutex
from queue import Queue, Empty
import time

class FrameProcessor(QThread):
    frame_processed = pyqtSignal(tuple)  # Сигнал с координатами пальца
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.running = True
        self.mutex = QMutex()
        self.async_mode = True  # По умолчанию используем асинхронный режим
        self.frame_queue = Queue(maxsize=2)  # Ограничиваем размер очереди до 2 кадров
        
    def run(self):
        while self.running:
            try:
                # Получаем кадр из очереди
                frame = self.frame_queue.get(timeout=0.1)  # Ждем кадр не более 0.1 секунды
                
                # Обрабатываем кадр
                x, y = self.model.get_finger(frame)
                if x > 0 and y > 0:
                    self.frame_processed.emit((x, y))
                    
            except Empty:
                # Если очередь пуста, просто продолжаем цикл
                continue
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
                    
    def add_frame(self, frame):
        if not self.running:
            return
            
        if self.async_mode:
            # В асинхронном режиме добавляем кадр в очередь
            try:
                # Если очередь полная, удаляем старый кадр
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                self.frame_queue.put_nowait(frame)
            except Exception:
                pass
        else:
            # В синхронном режиме обрабатываем кадр сразу
            try:
                x, y = self.model.get_finger(frame)
                if x > 0 and y > 0:
                    self.frame_processed.emit((x, y))
            except Exception as e:
                print(f"Error processing frame: {e}")
            
    def set_async_mode(self, enabled):
        """Включает или выключает асинхронный режим обработки"""
        self.async_mode = enabled
        # Очищаем очередь при смене режима
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
            
    def stop(self):
        self.running = False
        # Очищаем очередь перед остановкой
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        self.wait() 