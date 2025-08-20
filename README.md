# Gender Classification with ResNet50 + YOLOv8 + SORT + T5 based classification summary

This project provides an **end-to-end pipeline** for detecting people in video streams, tracking them across frames, and classifying gender using a fine-tuned ResNet-50. It combines **object detection (YOLOv8)**, **multi-object tracking (SORT)**, and **deep classification (ResNet50)** into a complete system.

---

## Features
- **Training**: Fine-tunes ResNet50 on custom gender-labeled datasets.
- **Detection**: Uses YOLOv8 for fast person detection in video frames.
- **Tracking**: Implements SORT (Kalman Filter + Hungarian matching) to assign persistent IDs to people.
- **Classification**: Cropped person images are classified as *Male* or *Female*.
- **Reporting**: Generates annotated output videos and a text summary of counts (with optional T5 natural-language generation).

---

## Project Structure
gender_classification/
├─ train.py # Train ResNet50 on gender dataset
├─ pipeline.py # End-to-end video pipeline (YOLOv8 + SORT + ResNet50)
├─ sort.py # SORT tracker implementation
├─ README.md # Project documentation (this file)
├─ .gitignore # Ignore rules for GitHub


---

## Requirements
Install the dependencies with pip:

```bash
pip install torch torchvision ultralytics scikit-learn transformers opencv-python tqdm filterpy


Usage
1. Train the Gender Classifier

1.1. Prepare a dataset with:
        Images (.jpg/.png)
        Labels (.txt files with format: gender: 0 or gender: 1)

1.2. Run training:
    python train.py
It saves the best model as best_resnet50_gender_model.pth
Logs metrics, accuracy/loss plots, and classification reports

2. Run the Video Pipeline
2.1. Provide an input video and trained model:
2.2. Run
    python pipeline.py
It outputs:
    Annotated video (bounding boxes, gender labels)
    Text report with gender counts
    Optional natural-language summary (using T5)


#Model & Data Notes
    ResNet50 is initialized with ImageNet weights and fine-tuned.
    Normalization is crucial: input images should match ImageNet mean/std.
    Labels: 0 = Male, 1 = Female (configurable).
    For imbalanced datasets, use class weights in training.

#Future Improvements
    Add “Unknown” class with confidence thresholding.
    Integrate live webcam mode.
    Export inference results as structured CSV/JSON.
    Extend attribute recognition (age, clothing, accessories).

Author
Sherzod Abdumalikov

AI Developer | Computer Vision & Deep Learning | Machine Learning | LLM | NLP
