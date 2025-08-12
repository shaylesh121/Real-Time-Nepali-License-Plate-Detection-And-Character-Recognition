import cv2
import os
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from datetime import datetime
import sqlite3
import shutil

# === Paths ===
# plate_model_path = '/mnt/c/Users/ayers/OneDrive/Desktop/project/model1/best.pt'
# char_model_path = '/mnt/c/Users/ayers/OneDrive/Desktop/project/model2(back)/best.pt'
# classifier_model_path = '/mnt/c/Users/ayers/OneDrive/Desktop/project/character_classifier.pth'

# for windows
plate_model_path = r'C:\Users\ayers\OneDrive\Desktop\project\model1\best.pt'
char_model_path = r'C:\Users\ayers\OneDrive\Desktop\project\model2(back)\best.pt'
classifier_model_path = r'C:\Users\ayers\OneDrive\Desktop\project\character_classifier.pth'


project_root = os.path.dirname(os.path.abspath(__file__))
top_output_dir = os.path.join(project_root, 'output', 'characters', 'top_row')
bottom_output_dir = os.path.join(project_root, 'output', 'characters', 'bottom_row')

# === Load YOLO Models ===
plate_model = YOLO(plate_model_path)
char_model = YOLO(char_model_path)

# === Character Classifier ===
def load_character_classifier(model_path):
    class_map = ['क', 'को', 'ख', 'ग', 'च', 'ज', 'झ', 'ञ', 'डि', 'त', 'ना', 'प', 'प्र', 'ब', 'बा', 'भे',
                 'म', 'मे', 'य', 'लु', 'सी', 'सु', 'से', 'ह', '०', '१', '२', '३', '४', '५', '६', '७', '८', '९']

    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_map))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    return model, class_map, transform

# === Predict Character ===
def predict_character(image_path, model, class_map, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    return class_map[pred.item()]

# === Extract Text from Directory ===
def extract_text_from_directory(dir_path, model, class_map, transform):
    files = sorted(os.listdir(dir_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
    return ''.join([predict_character(os.path.join(dir_path, f), model, class_map, transform) for f in files])

# === Save Cropped Characters ===
def save_cropped_characters(image, char_boxes):
    centers = np.array([[(y1 + y2) / 2] for (_, y1, _, y2) in char_boxes])
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(centers)
    labels = kmeans.labels_

    top_label = int(np.argmin([np.mean([c[0] for c, l in zip(centers, labels) if l == i]) for i in range(2)]))

    top_boxes = sorted([b for b, l in zip(char_boxes, labels) if l == top_label], key=lambda b: b[0])
    bottom_boxes = sorted([b for b, l in zip(char_boxes, labels) if l != top_label], key=lambda b: b[0])

    shutil.rmtree(top_output_dir, ignore_errors=True)
    shutil.rmtree(bottom_output_dir, ignore_errors=True)
    os.makedirs(top_output_dir, exist_ok=True)
    os.makedirs(bottom_output_dir, exist_ok=True)

    for idx, (x1, y1, x2, y2) in enumerate(top_boxes):
        crop = image[y1:y2, x1:x2]
        if crop.size:
            cv2.imwrite(os.path.join(top_output_dir, f'top_{idx}.jpg'), crop)

    for idx, (x1, y1, x2, y2) in enumerate(bottom_boxes):
        crop = image[y1:y2, x1:x2]
        if crop.size:
            cv2.imwrite(os.path.join(bottom_output_dir, f'bottom_{idx}.jpg'), crop)

def initialize_database():
    conn = sqlite3.connect('parking_lot.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            fare REAL
        )
    ''')
    conn.commit()
    conn.close()


# === Log Entry/Exit in Database ===
def log_to_database(plate_number):
    conn = sqlite3.connect('parking_lot.db')
    cursor = conn.cursor()

    cursor.execute("SELECT id, entry_time FROM entries WHERE plate_number = ? AND exit_time IS NULL", (plate_number,))
    row = cursor.fetchone()

    if row is None:
        cursor.execute("INSERT INTO entries (plate_number, entry_time) VALUES (?, ?)",
                       (plate_number, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        print(f"🟢 Entry logged for {plate_number}")
    else:
        entry_time = datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
        exit_time = datetime.now()
        duration = (exit_time - entry_time).total_seconds() / 60
        fare = round(duration * 2, 2)

        cursor.execute("UPDATE entries SET exit_time = ?, fare = ? WHERE id = ?",
                       (exit_time.strftime('%Y-%m-%d %H:%M:%S'), fare, row[0]))
        print(f"🔴 Exit logged for {plate_number} | Duration: {duration:.2f} mins | Fare: Rs. {fare}")

    conn.commit()
    conn.close()

# === Real-Time Inference from Camera ===
def run_camera_inference():
    initialize_database()  # <-- Call here to make sure the table exists
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Unable to access the webcam.")
        return

    model, class_map, transform = load_character_classifier(classifier_model_path)
    previous_text = ""
    stable_results = []
    same_count = 0
    MIN_STABLE_FRAMES = 3
    frame_id = 0

    print("🎥 Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % 10 != 0:
            continue

        results = plate_model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue

                plate_crop = frame[y1:y2, x1:x2]
                char_results = char_model(plate_crop)
                char_boxes = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in char_results[0].boxes.xyxy]

                if not char_boxes:
                    continue

                save_cropped_characters(plate_crop, char_boxes)
                top_text = extract_text_from_directory(top_output_dir, model, class_map, transform)
                bottom_text = extract_text_from_directory(bottom_output_dir, model, class_map, transform)
                full_text = top_text + bottom_text

                if full_text == previous_text:
                    same_count += 1
                else:
                    same_count = 1
                    previous_text = full_text

                if same_count >= MIN_STABLE_FRAMES and full_text not in stable_results:
                    print(f"[{frame_id}] ✅ Stable License Plate ➜", full_text)
                    log_to_database(full_text)
                    stable_results.append(full_text)

                break

        cv2.imshow("License Plate Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam stream ended.")

# === Run Entry Point ===
if __name__ == "__main__":
    run_camera_inference()
