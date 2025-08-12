# REAL TIME VEHICLE LICENSE PLATE DETECTION AND CHARACTER RECOGNITION

This project uses computer vision and deep learning to detect vehicle entries and exits in a parking lot using YOLOv8, clustering, and audio feedback.

---

## Features

- Real-time object detection with YOLOv8
- Vehicle clustering using KMeans
- Audio alerts for entry and exit
- Logging events with timestamps in SQLite
- Feature extraction using pretrained ResNet
- Modular and extensible design

---

## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up the database**
    - If you don't have an existing `parking_lot.db`, you can create one using:
      ```bash
      python create_db.py
      ```
      *(You’ll need to make this script or run the SQL manually)*

---

## Usage

Run the main system:
```bash
python main.py
```
```bash
dashboard.py
```
*(You'll need to run these two parallely( in two terminal) for the Vehicle number detection and observe the detected number plate in dashboard)

---

## Screenshots

1. First Detection of Licence Plate:
<img width="541" height="535" alt="image" src="https://github.com/user-attachments/assets/0b4ec55c-9cbc-41ea-aba9-01b0312b2b52" />


---


2. Second Detection of Character in Licence Plate:
<img width="676" height="392" alt="image" src="https://github.com/user-attachments/assets/a7d2574a-7d5d-48f4-bcf6-1f573dd69e4a" />


---


3. Recognization of Character in Licence Plate:
<img width="900" height="363" alt="image" src="https://github.com/user-attachments/assets/f89d742b-a3be-4c21-a078-a4a896bad1df" />
