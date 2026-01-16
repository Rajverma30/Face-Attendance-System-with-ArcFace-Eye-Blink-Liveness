# Face Attendance System with ArcFace + Eye Blink Liveness

A real-time **Face Recognition Attendance System** using **ArcFace** for high-accuracy face embeddings, enhanced with **Eye Blink Liveness Detection** to prevent spoofing.

Attendance is logged in **SQLite**, with future support for other databases.

---

##  Key Features

-  ArcFace-based Face Recognition (high accuracy)
-  Eye Blink Liveness Detection to prevent fake attendance
-  Real-time face detection and recognition
-  SQLite database for storing attendance
-  Date & time-based attendance logging
-  Modular architecture for future extensions

---

##  Project Structure

Fatttendance/
├── main.py
├── models/
│ └── antelopev2/
│ └── arcface.onnx (download separately, not in repo)
├── database.db
├── requirements.txt
└── README.md

yaml
Copy code

> Note: The ArcFace ONNX model is **not included in GitHub** due to file size limits. See instructions below to download it.

---

##  Setup Instructions

1. Clone the repo:
```bash
git clone https://github.com/Rajverma30/Face-Attendance-System-with-ArcFace-Eye-Blink-Liveness.git
cd Face-Attendance-System-with-ArcFace-Eye-Blink-Liveness
Install requirements:

bash
Copy code
pip install -r requirements.txt
Download ArcFace ONNX model and place it here:

makefile
Copy code
S:\Machine learning\Fatttendance\models\antelopev2\arcface.onnx
Run the project:

bash
Copy code
python main.py
Database
Current: SQLite

Future: PostgreSQL, MySQL, Cloud DB

Why Liveness Detection
Prevents fake attendance via photos or videos using:

Eye Blink Detection

Real-time verification

No blink = no attendance.

Use Cases
College / University attendance

Office employee attendance

Secure access control systems

Exam proctoring

Author
Raj
AI / ML Engineer
Focused on real-time, production-grade AI systems.

yaml
Copy code

---

## 2️ Download ArcFace ONNX model safely

Since GitHub **cannot host >100 MB**, do this:

1. **Download the ONNX model**  
You can get ArcFace ONNX here (official InsightFace / GitHub releases):  
[ArcFace ONNX model](https://github.com/deepinsight/insightface/releases)  
*(Pick the correct version, e.g., AntelopeV2 ONNX)*

2. **Place the model**  
Move the downloaded file here: 