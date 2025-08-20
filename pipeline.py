import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from ultralytics import YOLO
from collections import Counter
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sort import Sort



#Load_trained_gender_classification_model

from torchvision.models import ResNet50_Weights
from torch import nn
from torchvision import models

class ResNet50WithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 2)
        )

    def forward(self, x):
        return self.base(x)


#device_model_loading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gender_model = ResNet50WithDropout().to(device)
gender_model.load_state_dict(torch.load("best_resnet50_gender_model.pth", map_location=device))
gender_model.eval()


#image_transform

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


#YOLOv8_and_SORT_Init

yolo_model = YOLO("yolov8n.pt")  # or yolov8s.pt
tracker = Sort()


#Video_Input_and_Output_Setup

video_path = r"C:\Users\sherz\malgnst\gender_classification\test_video_1.mp4"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("gender_output_1.mp4", fourcc, fps, (width, height))

frame_index = 0
results_log = []


#Process Video Frame-by-Frame

print("[INFO] Processing video...")
seen_track_ids = set()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    yolo_results = yolo_model(frame)[0]

    # Prepare detections for SORT
    detections = []
    for box in yolo_results.boxes:
        cls_id = int(box.cls.item())
        if cls_id != 0:
            continue
        x1, y1, x2, y2 = box.xyxy[0]
        conf = float(box.conf[0])
        detections.append([x1.item(), y1.item(), x2.item(), y2.item(), conf])

    tracked_objects = tracker.update(np.array(detections))

    for track in tracked_objects:
        
        x1, y1, x2, y2, track_id = map(int, track[:5])
        cropped = frame[y1:y2, x1:x2]

        try:
            image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            image = resnet_transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = gender_model(image)
                pred = torch.argmax(output, dim=1).item()
                gender = "Male" if pred == 0 else "Female"

            results_log.append({
                "frame": frame_index,
                "id": track_id,
                "bbox": [x1, y1, x2, y2],
                "gender": gender
            })

            label = f"{gender} #{track_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            out.write(frame)

        except Exception as e:
            print(f"[WARNING] Skipped ID {track_id} due to error: {e}")

    frame_index += 1

cap.release()
out.release()


#Generate_LLM_based_summary_function

def generate_llm_summary(male_count, female_count, total_count, frames_tracked):
    prompt = (
        f"Video summary:\n"
        f"{total_count} people appeared in the video.\n"
        f"{male_count} were male and {female_count} were female.\n"
        f"Write a natural language summary report of this gender distribution."
    )

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

    input_ids = tokenizer.encode("summarize: " + prompt, return_tensors="pt").to(device)
    output_ids = model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return summary


#Filter_Duplicate IDs for Accurate Count

# Filter: Keep only the first appearance of each track ID
unique_results = {}
for r in results_log:
    track_id = r['id']
    if track_id not in unique_results:
        unique_results[track_id] = r

filtered_log = list(unique_results.values())


gender_counts = Counter([r['gender'] for r in filtered_log])
total = sum(gender_counts.values())
male = gender_counts.get("Male", 0)
female = gender_counts.get("Female", 0)




#Final Summary Report

report = (
    f" Person Gender Summary Report\n"
    f"Total People Detected: {total}\n"
    f"Male: {male}\n"
    f"Female: {female}\n"
)

print("\n" + "="*40)
print(report)
llm_summary = generate_llm_summary(male, female, total, frame_index)
print("\n LLM Summary:")
print(llm_summary)

# Save both summaries
with open("gender_report_llm.txt", "w") as f:
    f.write(report)
    f.write("\n\n--- LLM-Based Summary ---\n")
    f.write(llm_summary)

with open("gender_report.txt", "w") as f:
    f.write(report)

print("="*40) 