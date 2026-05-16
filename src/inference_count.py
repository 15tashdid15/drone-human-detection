
from ultralytics import YOLO
import cv2

HUMAN_CLASS = 0
CAR_CLASS = 1

def predict_and_count(model_path, image_path, save_path, conf=0.25):
    model = YOLO(model_path)

    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=conf,
        iou=0.5,
        verbose=False
    )

    result = results[0]
    img = cv2.imread(str(image_path))

    human_count = 0
    car_count = 0

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if cls_id == HUMAN_CLASS:
                human_count += 1
                label = f"human {score:.2f}"
                color = (0, 255, 0)
            elif cls_id == CAR_CLASS:
                car_count += 1
                label = f"car {score:.2f}"
                color = (255, 0, 0)
            else:
                continue

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                label,
                (x1, max(y1 - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    title = f"Total Humans: {human_count} | Cars: {car_count}"
    cv2.rectangle(img, (10, 10), (470, 55), (0, 0, 0), -1)
    cv2.putText(
        img,
        title,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imwrite(str(save_path), img)

    return human_count, car_count

if __name__ == "__main__":
    humans, cars = predict_and_count(
        model_path="training_run/weights/best.pt",
        image_path="sample.jpg",
        save_path="output.jpg"
    )

    print("Humans:", humans)
    print("Cars:", cars)
