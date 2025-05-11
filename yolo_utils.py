import numpy as np
from ultralytics import YOLO


def load_yolo(model_path='yolov8n.pt'):
    return YOLO(model_path)

def run_yolo(yolo_model, image):
    results = yolo_model(image, verbose=False)[0]
    bboxes = []
    classes = []
    confs = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        bboxes.append([x1, y1, x2, y2])
        classes.append(int(box.cls[0]))
        confs.append(float(box.conf[0]))
    return np.array(bboxes), np.array(classes), np.array(confs)
