
from ultralytics import YOLO

def train_model(data_yaml="visdrone_human_car_fast.yaml", epochs=10, imgsz=640, batch=16):
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        workers=2,
        pretrained=True,
        optimizer="auto",
        patience=3,
        close_mosaic=3,
        plots=True,
        val=True,
        cache=False
    )

    return results

if __name__ == "__main__":
    train_model()
