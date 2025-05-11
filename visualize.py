import cv2


def draw_detections(image, bboxes, classes, confidences, distances=None):
    for bbox, cls, conf, dist in zip(bboxes, classes, confidences, distances):
        x1, y1, x2, y2 = map(int, bbox)
        label = f'{cls} {conf:.2f}'
        if (distances is not None) and (dist is not None):
            label += f' | {dist:.1f} m'
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return image
