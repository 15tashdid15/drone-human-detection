
# Drone Human Detection and Counting System

## Project Overview

This project builds a computer vision pipeline for drone/aerial images using the VisDrone dataset.

The system can:
- Detect humans
- Detect cars
- Count total humans
- Draw bounding boxes
- Visualize final outputs
- Evaluate model performance

## Dataset

Dataset used: VisDrone aerial/drone image dataset.

The original VisDrone dataset contains 10 classes:

| Original ID | Class Name |
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

For this project, the original labels were converted into a focused 2-class dataset:

| New Class ID | New Class Name | Original Class |
|---:|---|---|
| 0 | human | pedestrian + people |
| 1 | car | car |

This mapping was used because the assessment focuses on human and car detection with human counting.

## Dataset Preparation

Due to Colab runtime and Google Drive file-access limitations, a smaller focused dataset was created from the original VisDrone dataset.

Final training subset:
- Training images: 500
- Validation images: 120
- Classes: 2
- Class 0: human
- Class 1: car

Only images containing human or car labels were selected. Other classes were ignored.

## Dataset Challenges

The main challenges noticed in the dataset were:
- Drone images contain very small objects.
- Humans are often tiny, crowded, or partially occluded.
- Cars appear from top-view and angled perspectives.
- Backgrounds are complex, including roads, buildings, trees, shadows, and crowded areas.
- Object scale varies greatly between images.

## Model

The model used in this project is YOLOv8n.

YOLOv8n was selected because it is lightweight, fast, and suitable for training on Google Colab T4 GPU.

## Training Setup

| Parameter | Value |
|---|---|
| Model | YOLOv8n |
| Image size | 640 |
| Epochs | 10 |
| Batch size | 16 |
| Device | Google Colab T4 GPU |
| Classes | human, car |

Best model path:

    /content/drive/MyDrive/internship_project/antlings_final_run_20260516_061807/training_run/weights/best.pt
    public link: https://drive.google.com/file/d/1Dqlh1Z4AtIYQd6gFf_3PqpnRl2qNvCu7/view?usp=sharing

## Preprocessing and Augmentation

The YOLO/Ultralytics training pipeline automatically applies preprocessing and augmentation, including:
- Image resizing
- Normalization
- Mosaic augmentation
- Scale augmentation
- HSV/color augmentation
- Horizontal flipping

## Counting Logic

Human count = number of detected boxes with class ID 0  
Car count = number of detected boxes with class ID 1

The output image displays:

    Total Humans: X | Cars: Y

## Inference and Visualization

During inference:
1. The trained YOLOv8n model predicts bounding boxes.
2. Human detections are drawn with labels.
3. Car detections are drawn with labels.
4. Total human count and car count are displayed on the image.
5. Final processed images are saved in the output folder.

## Evaluation

The model was evaluated on the validation set.

Evaluation outputs include:
- Precision
- Recall
- mAP50
- mAP50-95
- Confusion matrix
- PR curve
- F1 curve
- Validation prediction samples

## Output Files

Important output folders/files:
- outputs/human_car_counting/
- outputs/final_detection_grid.png
- outputs/counting_summary.csv
- outputs/demo_video/human_car_detection_demo.mp4
- training_run/
- validation_run/
- task01_dataset_understanding/

## Strengths

- Lightweight and fast detection pipeline.
- Focused on project-specific classes.
- Works on drone/aerial images.
- Counts humans automatically.
- Produces visual outputs with bounding boxes and count text.
- Practical for Colab T4 GPU within limited time.

## Limitations

- Very small humans may be missed.
- Crowded scenes can reduce detection accuracy.
- Occlusion can cause missed detections.
- The model was trained on a limited subset due to time and runtime limitations.
- More training images, more epochs, or a larger YOLO model could improve performance.

## Future Improvements

Possible improvements include:
- Train on the full VisDrone dataset.
- Use a larger model such as YOLOv8s or YOLOv8m.
- Add object tracking using ByteTrack or DeepSORT.
- Improve counting stability in video using tracking IDs.
- Tune confidence threshold and IoU threshold.
- Add FPS measurement for real-time performance analysis.

## How to Run

1. Install requirements.
2. Prepare the focused 2-class human/car dataset.
3. Train YOLOv8n.
4. Validate the trained model.
5. Run inference on validation/test images.
6. Save final bounding box and counting visualizations.

## Conclusion

This project successfully implements a drone-based human and car detection pipeline. The system detects humans and cars, counts total humans, and visualizes results with bounding boxes and count text.
