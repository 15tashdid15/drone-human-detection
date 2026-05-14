# Drone Human Detection and Counting System

This project is part of the Antlings Internship AI/ML technical assessment.

## Project Goal

The goal is to build a computer vision pipeline for drone/aerial images that can:

- Detect humans
- Detect cars
- Count total humans
- Visualize final outputs with bounding boxes and count overlay
- Evaluate the model performance
- Optionally perform object tracking

## Dataset

Dataset: VisDrone Dataset

The original VisDrone dataset contains multiple object classes. For this assessment, the project focuses on two target classes:

- Human
- Car

Pedestrian and people classes are merged into the human class.

## Project Structure

```text
src/
  prepare_dataset.py
  train.py
  detect_and_count.py
  evaluate.py
  track.py

docs/
  REPORT.md
  DEMO_VIDEO_SCRIPT.md

notebooks/
assets/
scripts/