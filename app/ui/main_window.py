from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                            QLabel, QFileDialog, QMessageBox, QMenuBar, QMenu, QAction)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import cv2
from models.yolo_small_model import YOLOSmallModel
from models.yolo_big_model import YOLOBigModel
from models.mediapipe_model import MediaPipeModel
from .frame_processor import FrameProcessor
from .gesture_processor import GestureProcessor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Finger Tracking")
        self.setGeometry(100, 100, 1200, 800)
        
        # Создаем центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Создаем главный layout
        main_layout = QVBoxLayout(central_widget)
        
        # Создаем меню
        self.create_menu()
        
        # Создаем область для отображения видео
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.mousePressEvent = self.on_video_label_click
        main_layout.addWidget(self.video_label)
        
        # Инициализируем переменные
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_model = None
        self.frame_processor = None
        self.gesture_processor = None  # Инициализируем как None
        self.gesture_enabled = False  # Флаг для включения/выключения жестов
        self.current_camera = 0
        self.points = []
        self.async_mode = True  # По умолчанию используем асинхронный режим
        
        # Создаем экземпляры моделей
        self.models = {
            "YOLO Small": YOLOSmallModel(),
            "YOLO Big": YOLOBigModel(),
            "MediaPipe": MediaPipeModel()
        }
        
        # Устанавливаем начальную модель
        self.set_model("YOLO Small")
        
        # Запускаем камеру при старте
        self.start_camera()
        
    def create_menu(self):
        menubar = self.menuBar()
        
        # Меню File
        file_menu = menubar.addMenu("File")
        
        # Добавляем действия в меню File
        open_action = QAction("Open Video", self)
        open_action.triggered.connect(self.open_video)
        file_menu.addAction(open_action)
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Меню Camera
        camera_menu = menubar.addMenu("Camera")
        
        # Добавляем действия в меню Camera
        for camera_id in ["0", "1", "2"]:
            action = QAction(f"Camera {camera_id}", self)
            action.triggered.connect(lambda checked, val=camera_id: self.set_camera(val))
            camera_menu.addAction(action)
            
        # Меню Model
        model_menu = menubar.addMenu("Model")
        
        # Добавляем действия в меню Model
        for model_name in ["YOLO Small", "YOLO Big", "MediaPipe"]:
            action = QAction(model_name, self)
            action.triggered.connect(lambda checked, val=model_name: self.set_model(val))
            model_menu.addAction(action)
        
        # Меню Control
        control_menu = menubar.addMenu("Control")
        
        # Добавляем действия в меню Control
        start_action = QAction("Start", self)
        start_action.triggered.connect(self.start_camera)
        control_menu.addAction(start_action)
        
        stop_action = QAction("Stop", self)
        stop_action.triggered.connect(self.stop_camera)
        control_menu.addAction(stop_action)
        
        # Добавляем действие очистки
        clear_action = QAction("Clear Drawing", self)
        clear_action.triggered.connect(self.clear_drawing)
        control_menu.addAction(clear_action)
        
        # Добавляем действие включения/выключения жестов
        self.gesture_action = QAction("Enable Gesture Control", self)
        self.gesture_action.setCheckable(True)  # Делаем действие переключаемым
        self.gesture_action.triggered.connect(self.toggle_gesture_control)
        control_menu.addAction(self.gesture_action)
        
        # Добавляем действие включения/выключения асинхронного режима
        self.async_action = QAction("Enable Async Processing", self)
        self.async_action.setCheckable(True)  # Делаем действие переключаемым
        self.async_action.setChecked(True)  # По умолчанию включено
        self.async_action.triggered.connect(self.toggle_async_mode)
        control_menu.addAction(self.async_action)
        
    def toggle_async_mode(self, checked):
        """Включает или выключает асинхронный режим обработки"""
        self.async_mode = checked
        if self.frame_processor:
            self.frame_processor.set_async_mode(checked)
        print(f"Async mode {'enabled' if checked else 'disabled'}")
        
    def toggle_gesture_control(self, checked):
        """Включает или выключает распознавание жестов"""
        self.gesture_enabled = checked
        if checked:
            if self.gesture_processor is None:
                self.gesture_processor = GestureProcessor()
                self.gesture_processor.gesture_detected.connect(self.on_gesture_detected)
                self.gesture_processor.start()
        else:
            if self.gesture_processor is not None:
                self.gesture_processor.stop()
                self.gesture_processor = None
                
    def clear_drawing(self):
        self.points = []
        print("Drawing cleared")
        
    def set_model(self, model_name):
        if self.frame_processor:
            self.frame_processor.stop()
            self.frame_processor = None
            
        if self.current_model:
            self.current_model.cleanup()
            
        # Пересоздаем модель при переключении
        if model_name == "YOLO Small":
            self.models[model_name] = YOLOSmallModel()
        elif model_name == "YOLO Big":
            self.models[model_name] = YOLOBigModel()
        elif model_name == "MediaPipe":
            self.models[model_name] = MediaPipeModel()
            
        self.current_model = self.models[model_name]
        self.points = []  # Очищаем точки при смене модели
        
        # Создаем новый процессор кадров
        self.frame_processor = FrameProcessor(self.current_model)
        self.frame_processor.frame_processed.connect(self.on_frame_processed)
        self.frame_processor.set_async_mode(self.async_mode)  # Устанавливаем текущий режим
        self.frame_processor.start()
        
        print(f"Model set to: {model_name}")
        
    def on_frame_processed(self, coords):
        x, y = coords
        if not self.points:  # Если это первая точка
            self.points = [(x, y)]
        else:
            self.points.append((x, y))
        
    def set_camera(self, camera_id):
        self.current_camera = int(camera_id)
        if self.cap is not None:
            self.cap.release()
            self.cap = cv2.VideoCapture(self.current_camera)
        print(f"Camera set to: {camera_id}")
        
    def start_camera(self):
        if not self.timer.isActive():
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.current_camera)
            if self.cap.isOpened():
                self.timer.start(30)  # 30 FPS
                print("Camera started")
            else:
                QMessageBox.critical(self, "Error", "Could not open camera")
                
    def stop_camera(self):
        if self.timer.isActive():
            self.timer.stop()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            print("Camera stopped")
                
    def open_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_name:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(file_name)
            if self.cap.isOpened():
                self.timer.start(30)
                print(f"Video opened: {file_name}")
            else:
                QMessageBox.critical(self, "Error", "Could not open video file")
                
    def on_gesture_detected(self, is_fist):
        """Обработчик обнаружения жеста"""
        if is_fist:
            self.clear_drawing()
            
    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return
            
        # Отправляем кадр на обработку
        if self.frame_processor:
            self.frame_processor.add_frame(frame)
            
        # Отправляем кадр на обработку жестов только если они включены
        if self.gesture_enabled and self.gesture_processor:
            self.gesture_processor.add_frame(frame)
            
        # Отображаем кадр
        self.display_frame(frame)
        
    def display_frame(self, frame):
        # Конвертируем кадр в формат для отображения
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Создаем QPixmap и рисуем на нем
        pixmap = QPixmap.fromImage(qt_image)
        painter = QPainter(pixmap)
        pen = QPen(Qt.red, 3)
        painter.setPen(pen)
        
        # Рисуем линию
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                painter.drawLine(
                    int(self.points[i][0]), int(self.points[i][1]),
                    int(self.points[i+1][0]), int(self.points[i+1][1])
                )
        
        # Рисуем круг на последней позиции пальца
        if self.points:
            last_point = self.points[-1]
            painter.drawEllipse(int(last_point[0]) - 10, int(last_point[1]) - 10, 20, 20)
        
        painter.end()
        
        # Масштабируем изображение под размер окна
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        
    def closeEvent(self, event):
        if self.frame_processor:
            self.frame_processor.stop()
        if self.gesture_processor:
            self.gesture_processor.stop()
        if self.current_model:
            self.current_model.cleanup()
        if self.cap is not None:
            self.cap.release()
        event.accept()

    def on_video_label_click(self, event):
        """Обработчик клика по видео"""
        self.clear_drawing() 