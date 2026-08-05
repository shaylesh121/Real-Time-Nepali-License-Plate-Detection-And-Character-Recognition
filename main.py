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
import pygame
pygame.mixer.init()

# === Paths ===
plate_model_path = r'C:\Users\saile\OneDrive\Desktop\project\model1\best.pt'
char_model_path = r'C:\Users\saile\OneDrive\Desktop\project\model2(back)\best.pt'
classifier_model_path = r'C:\Users\saile\OneDrive\Desktop\project\character_classifier.pth'

top_output_dir = os.path.join('output', 'characters', 'top_row')
bottom_output_dir = os.path.join('output', 'characters', 'bottom_row')

# === Load Models ===
plate_model = YOLO(plate_model_path)
char_model = YOLO(char_model_path)

def load_character_classifier(model_path):
    class_map = ['\u0915', '\u0915\u094b', '\u0916', '\u0917', '\u091a', '\u091c', '\u091d', '\u091e', '\u0921\u093f',
                 '\u0924', '\u0928\u093e', '\u092a', '\u092a\u094d\u0930', '\u092c', '\u092c\u093e', '\u092d\u0947',
                 '\u092e', '\u092e\u0947', '\u092f', '\u0932\u0941', '\u0938\u0940', '\u0938\u0941', '\u0938\u0947', '\u0939',
                 '०', '१', '२', '३', '४', '५', '६', '७', '८', '९']
  #  list of 34 characters here
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_map))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    return model, class_map, transform

def play_sound(path):
    try:
        sound = pygame.mixer.Sound(path)
        sound.play()
    except Exception as e:
        print(f"❌ Failed to play sound: {e}")


def predict_character(image_path, model, class_map, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    return class_map[pred.item()]

def extract_text_from_directory(dir_path, model, class_map, transform):
    files = sorted(os.listdir(dir_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
    return ''.join([predict_character(os.path.join(dir_path, f), model, class_map, transform) for f in files])

def save_cropped_characters(image, char_boxes):
    if len(char_boxes) < 2:
        return
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
        cv2.imwrite(os.path.join(top_output_dir, f'top_{idx}.jpg'), crop)
    for idx, (x1, y1, x2, y2) in enumerate(bottom_boxes):
        crop = image[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(bottom_output_dir, f'bottom_{idx}.jpg'), crop)

def parse_entry_time(entry_time):
    if isinstance(entry_time, datetime): return entry_time
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
        try: return datetime.strptime(entry_time, fmt)
        except ValueError: continue
    raise ValueError(f"Invalid time: {entry_time}")

def log_to_database(plate_number, is_exit=False, entry_time=None):
    with sqlite3.connect('parking_lot.db', timeout=10) as conn:
        cursor = conn.cursor()
        now = datetime.now()
        if not is_exit:
            cursor.execute("INSERT INTO entries (plate_number, entry_time) VALUES (?, ?)", (plate_number, now))
            print(f"\U0001F7E2 Entry: {plate_number}")
            play_sound("entry.wav")
        else:
            duration = (now - entry_time).total_seconds()
            fare = max(20, ((int(duration) + 19) // 20) * 20)
            cursor.execute("UPDATE entries SET exit_time = ?, fare = ? WHERE plate_number = ? AND exit_time IS NULL", (now.strftime('%Y-%m-%d %H:%M:%S'), fare, plate_number))
            print(f"\U0001F534 Exit: {plate_number} | Time: {int(duration)}s | Fare: Rs.{fare}")
            play_sound("exit.wav")
        conn.commit()

def run_camera_inference():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No camera")
        return

    model, class_map, transform = load_character_classifier(classifier_model_path)
    FRAME_SKIP, MIN_STABLE_FRAMES, frame_id = 10, 3, 0
    previous_text, same_count = '', 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            cv2.imshow("LPR", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        current_time = datetime.now()
        results = plate_model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if float(box.conf[0]) < 0.5:
                    continue

                plate_crop = frame[y1:y2, x1:x2]
                char_results = char_model(plate_crop)
                char_boxes = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in char_results[0].boxes.xyxy]
                if len(char_boxes) < 2: continue

                save_cropped_characters(plate_crop, char_boxes)
                top_text = extract_text_from_directory(top_output_dir, model, class_map, transform)
                bottom_text = extract_text_from_directory(bottom_output_dir, model, class_map, transform)
                full_text = top_text + bottom_text

                if full_text == previous_text:
                    same_count += 1
                else:
                    same_count = 1
                    previous_text = full_text

                if same_count >= MIN_STABLE_FRAMES:
                    with sqlite3.connect('parking_lot.db') as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT entry_time FROM entries WHERE plate_number = ? AND exit_time IS NULL", (full_text,))
                        row = cursor.fetchone()

                        if row:
                            entry_time = parse_entry_time(row[0])
                            elapsed = (current_time - entry_time).total_seconds()
                            if elapsed >= 10:
                                log_to_database(full_text, is_exit=True, entry_time=entry_time)
                        else:
                            cursor.execute("SELECT exit_time FROM entries WHERE plate_number = ? ORDER BY exit_time DESC LIMIT 1", (full_text,))
                            exit_row = cursor.fetchone()
                            if exit_row and exit_row[0]:
                                last_exit = parse_entry_time(exit_row[0])
                                elapsed = (current_time - last_exit).total_seconds()
                                if elapsed >= 10:
                                    log_to_database(full_text, is_exit=False)
                            else:
                                log_to_database(full_text, is_exit=False)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, full_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    break

        cv2.imshow("LPR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera_inference()