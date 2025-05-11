from ultralytics import YOLO
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to yaml file")
    parser.add_argument("--model", type=str, default='yolov9s.pt', help="Model version format yolo<version>.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Num of batches")
    parser.add_argument("--workers", type=int, default=10, help="Num of workers")
    parser.add_argument("--epochs", type=int, default=50, help="Train epochs count")
    parser.add_argument('--project', type=str, default='project')

    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                          device="cuda", project=args.project)