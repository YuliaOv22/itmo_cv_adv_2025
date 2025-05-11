# Домашнее задание №6. Виртуальная доска
# Отчет о проделанной работе
###### Команда #1 (Никитина Алина, Овчинникова Юлия, Горохова Александра)  
  
## 1. Обучение модели определения ключевых точек руки

### 1.1. Датасет

Для обучения модели определения ключевых точек руки был выбран датасет [Hand Keypoint Dataset 26K](https://www.kaggle.com/datasets/riondsilva21/hand-keypoint-dataset-26k)

Датасет состоит из следующих частей:

**images**: Изображения, разделенные на train и val.  
**annotations**: Аннотации для изображений в формате COCO.  
**labels**: Аннотации для изображений в формате YOLO.  

Каждая аннотация содержит в себе:
- координаты bounding box для рук на изображении
- координаты keypoints для руки

Ключевые точки аннотируются следующим образом:
- Запястье
- Большой палец (4 точки)
- Указательный палец (4 точки)
- Средний палец (4 точки)
- Безымянный палец (4 точки)
- Мизинец (4 точки)
  
Каждая рука имеет в общей сложности 21 ключевую точку.

![image](https://github.com/user-attachments/assets/bdea2d1f-b538-4a75-9e03-11f389b490e5)

### 1.2. Обучение модели

В ходе работы для данной задачи были обучены 2 модели YOLO: **yolo11n-pose** и **yolo11x-pose**. Файл конфигурации data.yaml [тут](). Файл обучения модели [тут]().
  
Keypoints в датасете представлены в формате [21, 3] - 21 ключевая точка, каждая состоит из координат x,y и значения visible.

[Графики и метрики обучения модели YOLO11n-pose](https://github.com/YuliaOv22/itmo_cv_adv_2025/blob/lab_6/logs_yolo11n.md)  
[Графики и метрики обучения модели YOLO11x-pose](https://github.com/YuliaOv22/itmo_cv_adv_2025/blob/lab_6/logs_yolo11x.md)  

## 2. Сравнение с open-source моделями
Для сравнения были выбраны следующие модели, которые были найдены в open-source:
1. [YOLO11n](https://github.com/chrismuntean/YOLO11n-pose-hands) (модель, обученная по туториалу [Ultralytics](https://docs.ultralytics.com/datasets/pose/hand-keypoints/#what-are-the-key-features-of-the-hand-keypoints-dataset) и на таком же датасете)
2. [MediaPipe Hands](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html). Гайд по использованию был взят [тут](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)

*P.S. Также хотелось сравнить результаты с моделью из репозитория [Hand Keypoint Detection using Deep Learning and OpenCV](https://github.com/erezposner/MV_HandKeyPointDetector), но ссылка на веса данной модели оказалась не рабочей и найти ее не удалось*  

Сравнение моделей производилось на исходном датасете для валидации. Результаты, полученные от всех моделей были преобразованы в формат COCO, а затем оценены с помощью pycocotools COCOeval. [Jupyter Notebook с кодом inference моделей на валидационном датасете и оценкой метрик каждой из моделей]()

**Таблица сравнения моделей**

Примеры сравнения работы моделей YOLOv11x и MediaPipe на наиболее трудных для моделей примерах (там, где был показан наиболее плохой результат):

<table style="width: 100%; text-align: center;">
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/ef58757d-2a32-4092-a148-8efec01a59d9" alt="Image 1" style="width: 220px; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/83f15fe7-8ec8-4e6a-b79b-b56596a37a05" alt="Image 2" style="width: 220px; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/1ced2d2d-452a-479a-b273-9f67933903b6" alt="Image 1" style="width: 220px; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/050b90fa-6ad1-4e27-9c56-a1ebde85406f" alt="Image 2" style="width: 220px; height: auto;">
    </td>
  </tr>
  <tr>
    <td>yolo11x-pose</td>
    <td>mediapipe_hand</td>
    <td>yolo11x-pose</td>
    <td>mediapipe_hand</td>
  </tr>
</table>

<table style="width: 100%; text-align: center;">
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/6bff1f88-5a7f-46aa-a5de-0ae5853fbda7" alt="Image 1" style="width: 220px; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/1b7be141-57d4-4ff0-a981-5cc9269bff08" alt="Image 2" style="width: 220px; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/c4ef9dcf-5302-4cc7-bf7a-6d81a935db81" alt="Image 1" style="width: 220px; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/6c2815a0-546a-4d3e-8441-ccbac9ab4930" alt="Image 2" style="width: 220px; height: auto;">
    </td>
  </tr>
  <tr>
    <td>yolo11x-pose</td>
    <td>mediapipe_hand</td>
    <td>yolo11x-pose</td>
    <td>mediapipe_hand</td>
  </tr>
</table>

**Сравнение на изображениях не из исходного датасета**

Кроме того, был проведен анализ работы моделей на фото из открытых источников.

<table style="width: 100%; text-align: center;">
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/2f1cea1d-805f-4e69-b04e-57d204801396" alt="Image 1" style="width: 220px; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/f6e497cb-dbf7-44c9-bcbd-8cc0e0a2e0da" alt="Image 2" style="width: 220px; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/294e9302-9417-4b39-a94d-6222492aa41a" alt="Image 1" style="width: 220px; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/a48ad178-77a4-4eb6-b855-6dcc36ba26d0" alt="Image 2" style="width: 220px; height: auto;">
    </td>
  </tr>
  <tr>
    <td>yolo11x-pose</td>
    <td>mediapipe_hand</td>
    <td>yolo11x-pose</td>
    <td>mediapipe_hand</td>
  </tr>
</table>

<table style="width: 100%; text-align: center;">
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/40745b09-283e-4755-b6bc-051d381816f4" alt="Image 1" style="width: 220px; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/7f81cb50-74e0-4290-b21a-33a3e468df39" alt="Image 2" style="width: 220px; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/8f4538e7-83ef-4245-a94d-12fb924f54bf" alt="Image 1" style="width: 220px; height: auto;">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/672ebfd4-e713-4b98-b7ec-512f062c1c68" alt="Image 2" style="width: 220px; height: auto;">
    </td>
  </tr>
  <tr>
    <td>yolo11x-pose</td>
    <td>mediapipe_hand</td>
    <td>yolo11x-pose</td>
    <td>mediapipe_hand</td>
  </tr>
</table>

На таких изображениях, часто MediaPipe Hand модель показывала более хороший результат.

## 3. Приложение "Виртуальная доска"

Следующей частью задания было написать приложение, которое осуществляет:

1. Считывание кадров с веб-камеры
2. Нахождение указательных пальцев, если они есть в кадре
3. Рисование на кадре траектории движения пальца
4. Вывод обработанных кадры в отдельное окно

В качестве фреймворка для интерфейса был использован PyQt5.
