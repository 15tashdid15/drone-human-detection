# Drone Human Detection and Counting System

## Project Overview

This project is a computer vision pipeline for detecting **humans** and **cars** from drone/aerial images using the **VisDrone** dataset. It was developed for the Antlings Internship AI/ML technical assessment.

The system performs:

- Dataset understanding and preprocessing
- Human and car detection
- Human counting
- Bounding box visualization
- Model training using YOLOv8
- Validation and evaluation
- Output image and demo video generation

---

## Project Goal

The goal is to build a drone/aerial image analysis system that can:

1. Detect humans in aerial images.
2. Detect cars in aerial images.
3. Count the total number of detected humans.
4. Display bounding boxes around detected objects.
5. Visualize the final processed outputs.
6. Evaluate the detection model.

---

## Dataset

The project uses the **VisDrone** aerial/drone image dataset.

The original VisDrone dataset contains 10 object classes:

| Original Class ID | Class Name |
|---:|---|
| 0 | pedestrian |
| 1 | people |
| 2 | bicycle |
| 3 | car |
| 4 | van |
| 5 | truck |
| 6 | tricycle |
| 7 | awning-tricycle |
| 8 | bus |
| 9 | motor |

The original dataset structure used in Colab was:

```text
VisDrone_Dataset/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── labels/
├── VisDrone2019-DET-val/
│   ├── images/
│   └── labels/
└── VisDrone2019-DET-test-dev/
    └── images/
```

---

## Class Mapping

The assessment focuses on **human** and **car** detection. Therefore, the original 10-class VisDrone labels were converted into a focused 2-class format.

| New Class ID | New Class Name | Original VisDrone Class |
|---:|---|---|
| 0 | human | pedestrian + people |
| 1 | car | car |

So the final detection classes are:

```text
0 = human
1 = car
```

This mapping directly matches the project requirement of detecting humans and cars and counting humans.

---

## Dataset Understanding

Before training, the dataset labels were analyzed to understand the class distribution.

Full training label analysis:

| Class | Bounding Box Count |
|---|---:|
| pedestrian | 79,385 |
| people | 27,067 |
| bicycle | 10,483 |
| car | 145,291 |
| van | 24,993 |
| truck | 12,910 |
| tricycle | 4,812 |
| awning-tricycle | 3,246 |
| bus | 5,933 |
| motor | 29,652 |

Total training bounding boxes:

```text
343,772
```

Important project classes in the full training set:

```text
Human-related boxes = pedestrian + people = 106,452
Car boxes = 145,291
```

Full validation label analysis:

| Class | Bounding Box Count |
|---|---:|
| pedestrian | 9,063 |
| people | 5,262 |
| bicycle | 1,321 |
| car | 14,814 |
| van | 2,048 |
| truck | 767 |
| tricycle | 1,069 |
| awning-tricycle | 533 |
| bus | 265 |
| motor | 4,991 |

Total validation bounding boxes:

```text
40,133
```

Important project classes in the full validation set:

```text
Human-related boxes = pedestrian + people = 14,325
Car boxes = 14,814
```

---

## Focused Training Subset

Due to Google Colab runtime limits and slow Google Drive file access, a smaller focused dataset was created from the original VisDrone dataset.

Final focused subset:

| Split | Images | Labels |
|---|---:|---:|
| Train | 500 | 500 |
| Validation | 120 | 120 |

Only images containing at least one **human** or **car** label were selected. Other classes were ignored because they are outside the assessment scope.

The prepared 2-class dataset structure was:

```text
visdrone_human_car_fast/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

## Dataset Challenges

The main challenges noticed in the dataset were:

- Drone images contain very small objects.
- Humans are often tiny in high-altitude aerial views.
- Crowded scenes make individual human detection difficult.
- Some humans and cars are partially occluded.
- Cars appear from top-view or angled perspectives.
- Backgrounds are complex, including roads, buildings, trees, shadows, and crowded areas.
- Object scale varies significantly between images.

---

## Preprocessing

The preprocessing pipeline included:

1. Reading original VisDrone YOLO-format labels.
2. Keeping only the classes needed for the project.
3. Mapping original class `0` pedestrian and class `1` people to new class `0` human.
4. Mapping original class `3` car to new class `1` car.
5. Ignoring all other classes.
6. Copying selected images into a clean local Colab dataset folder.
7. Saving new 2-class YOLO label files.

The bounding box coordinates were kept unchanged because the original labels were already in YOLO format.

---

## Augmentation

The YOLO/Ultralytics training pipeline automatically applied augmentation and preprocessing steps such as:

- Image resizing to 640
- Normalization
- Mosaic augmentation
- Scale augmentation
- HSV/color augmentation
- Horizontal flipping

These augmentations help the model generalize better to drone images with different object sizes, lighting conditions, and perspectives.

---

## Model

The model used in this project is:

```text
YOLOv8n
```

YOLOv8n was selected because:

- It is lightweight.
- It trains quickly on Google Colab T4 GPU.
- It is suitable for object detection tasks.
- It allows fast inference and visualization.
- It is practical for a limited-time technical assessment.

---

## Training Setup

| Parameter | Value |
|---|---|
| Model | YOLOv8n |
| Pretrained Weights | yolov8n.pt |
| Image Size | 640 |
| Epochs | 10 |
| Batch Size | 16 |
| Device | Google Colab T4 GPU |
| Number of Classes | 2 |
| Classes | human, car |
| Optimizer | auto |
| Patience | 3 |
| close_mosaic | 3 |

The best trained model is saved at:

```text
training_run/weights/best.pt
```

---

## Counting Logic

The counting logic is simple and direct.

```text
human_count = number of detected boxes with class ID 0
car_count = number of detected boxes with class ID 1
```

Each processed image displays a count label similar to:

```text
Total Humans: 12 | Cars: 7
```

The exact values change depending on the input image and model predictions.

---

## Inference and Visualization

During inference:

1. The trained YOLOv8n model predicts bounding boxes.
2. Human detections are labeled as `human`.
3. Car detections are labeled as `car`.
4. Human and car counts are calculated.
5. The total count text is displayed on the image.
6. The processed image is saved to the output folder.

Final detection outputs are saved in:

```text
outputs/human_car_counting/
```

A combined visualization grid is saved as:

```text
outputs/final_detection_grid.png
```

The counting summary is saved as:

```text
outputs/counting_summary.csv
```

A short generated demo video is saved as:

```text
outputs/demo_video/human_car_detection_demo.mp4
```

---

## Evaluation

The trained model was evaluated on the validation subset.

Evaluation outputs are saved in:

```text
validation_run/
```

The validation folder contains Ultralytics-generated evaluation plots and metrics such as:

- Precision
- Recall
- mAP50
- mAP50-95
- Confusion matrix
- PR curve
- F1 curve
- Validation prediction samples

No manual metric values are hardcoded in this README. The latest generated results should be checked inside the `validation_run/` folder.

---

## Repository Structure

```text
drone-human-detection/
├── README.md
├── requirements.txt
├── visdrone_human_car_fast.yaml
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── inference_count.py
├── task01_dataset_understanding/
│   └── sample_dataset_visualization.png
├── training_run/
│   ├── weights/
│   │   └── best.pt
│   └── results and training plots
├── validation_run/
│   └── validation metrics and plots
└── outputs/
    ├── human_car_counting/
    ├── counting_summary.csv
    ├── final_detection_grid.png
    └── demo_video/
        └── human_car_detection_demo.mp4
```

---

## Important Files

| File/Folder | Purpose |
|---|---|
| `README.md` | Full project explanation |
| `requirements.txt` | Python dependencies |
| `visdrone_human_car_fast.yaml` | YOLO dataset configuration |
| `src/preprocess.py` | Converts VisDrone labels into human/car labels |
| `src/train.py` | YOLOv8 training script |
| `src/inference_count.py` | Inference and counting script |
| `training_run/weights/best.pt` | Best trained YOLO model |
| `validation_run/` | Evaluation outputs |
| `outputs/` | Final detection, counting, and demo outputs |

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/15tashdid15/drone-human-detection.git
cd drone-human-detection
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Place the original VisDrone dataset outside or inside the project folder using this structure:

```text
VisDrone_Dataset/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── labels/
└── VisDrone2019-DET-val/
    ├── images/
    └── labels/
```

Then use the preprocessing function in `src/preprocess.py` to create the focused human/car dataset.

Example:

```python
from src.preprocess import convert_visdrone_to_human_car

convert_visdrone_to_human_car(
    src_root="VisDrone_Dataset",
    dst_root="visdrone_human_car_fast",
    train_limit=500,
    val_limit=120
)
```

### 4. Update Dataset YAML if Needed

The dataset YAML should point to the prepared 2-class dataset.

Example:

```yaml
path: ./visdrone_human_car_fast
train: images/train
val: images/val
nc: 2
names:
  0: human
  1: car
```

### 5. Train the Model

```bash
python src/train.py
```

The best model will be saved under the training run folder:

```text
training_run/weights/best.pt
```

### 6. Run Inference and Counting

Use the trained model for inference on drone images.

Example usage from Python:

```python
from src.inference_count import predict_and_count

humans, cars = predict_and_count(
    model_path="training_run/weights/best.pt",
    image_path="sample.jpg",
    save_path="output.jpg",
    conf=0.25
)

print("Humans:", humans)
print("Cars:", cars)
```

---

## Results

The project successfully produced:

- Human and car detection outputs
- Human counting visualization
- Processed output images
- Counting summary CSV
- Validation plots and metrics
- Demo video output
- Trained YOLOv8 model weights

---

## Strengths

- Complete end-to-end detection pipeline.
- Focused on the exact required classes: human and car.
- Uses YOLOv8n for fast training and inference.
- Includes human counting logic.
- Saves visual output images with bounding boxes and counts.
- Includes validation results and output artifacts.
- Practical for Google Colab T4 GPU.

---

## Limitations

- The model was trained on a limited subset due to runtime and Drive-access constraints.
- Very small humans may be missed in high-altitude drone images.
- Crowded scenes can reduce counting accuracy.
- Occlusion can cause missed detections.
- The model may confuse small or partially visible objects.
- Full-dataset training would likely improve performance.
- Object tracking was not implemented in the final version.

---

## Future Improvements

Possible future improvements include:

- Train on the full VisDrone dataset.
- Increase the number of epochs.
- Use a larger YOLO model such as YOLOv8s or YOLOv8m.
- Add object tracking using ByteTrack or DeepSORT.
- Use tracking IDs to improve counting stability in videos.
- Tune confidence and IoU thresholds.
- Measure FPS for real-time performance analysis.
- Test on more drone videos and unseen aerial images.

---

## Demo Video

A demo video was generated to show detection and counting outputs.

Expected demo video path:

```text
outputs/demo_video/human_car_detection_demo.mp4
```

For final submission, the demo video can also be shared through a Google Drive folder link.

---

## Conclusion

This project implements a complete drone-based human and car detection pipeline. It converts the VisDrone dataset into a focused human/car detection task, trains a YOLOv8n model, evaluates the model, detects humans and cars, counts total humans, and visualizes final outputs.

Although the model was trained on a limited subset due to Colab time and file-access constraints, the project demonstrates the full required workflow: dataset understanding, preprocessing, training, inference, visualization, evaluation, and documentation.
