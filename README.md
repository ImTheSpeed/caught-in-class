
# CAUGHT IN CLASS – AI-Powered Attendance Monitoring System

**CAUGHT IN CLASS** is an AI-enhanced classroom attendance monitoring system designed to streamline and automate attendance tracking through real-time face detection and recognition. Leveraging deep learning with YOLO and face recognition algorithms, it eliminates manual roll calls, reduces human error, and generates accurate attendance reports for lectures.

## Overview

This project utilizes computer vision and machine learning to detect, recognize, and log student attendance in real-time. It combines object detection, face recognition, and structured data logging to ensure minimal human intervention and maximum efficiency.

## Key Features

- **YOLO-based Person Detection**  
  Uses the YOLOv8 model to identify the presence of individuals in the video stream with high accuracy and performance.

- **Face Recognition Matching**  
  Matches captured faces against pre-encoded face data to identify known students using the `face_recognition` library.

- **Real-Time Monitoring**  
  Captures and analyzes live video from a connected webcam to detect and track students during class sessions.

- **Lecture Detection Logic**  
  Implements logic to determine whether a lecture is in progress based on the number of detected individuals.

- **Automated Attendance Logging**  
  Records attendance data into an Excel file, including timestamp, number of students present at the start and end of class, individual student names, and lecture status.

## How It Works

1. **Initialization**  
   - Loads a pre-trained YOLOv8 model.  
   - Encodes known student faces from images stored in the `faces/` directory.

2. **Start-of-Lecture Detection**  
   - Begins by counting individuals using YOLO to estimate student attendance at the beginning of the session.

3. **Face Recognition**  
   - Continuously processes frames to recognize and identify students from the known face dataset.

4. **Lecture Validation**  
   - Confirms if a lecture is taking place by checking if a minimum number of people are detected.  
   - Displays real-time status: *Lecture In Progress* or *Not a Lecture*.

5. **Data Export**  
   - Logs the following data in `attendance_final_file.xlsx`:
     - Timestamp  
     - Number of students at the start of the lecture  
     - Names of recognized students  
     - Number of students at the end of the lecture  
     - Overall status

## Tech Stack

- **Programming Language:** Python
- **Libraries & Frameworks:**
  - `OpenCV` – Real-time video stream handling
  - `YOLOv8 (Ultralytics)` – Person detection
  - `face_recognition` – Facial feature encoding and recognition
  - `NumPy` – Numerical operations and matrix computations
  - `Pandas` – Structured data logging
  - `OpenPyXL` – Excel file generation

## Installation & Requirements

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/caught-in-class.git
   cd caught-in-class
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place known face images inside the `faces/` directory.  
   - Image filenames should correspond to student names (e.g., `john_doe.jpg`).

4. Run the program:
   ```bash
   python CAUGHT_IN_CLASS.py
   ```

## Output

- A real-time video interface displaying:
  - Detected persons and their confidence scores
  - Recognized student names
  - Lecture status
- An Excel file `attendance_final_file.xlsx` summarizing the session

## Project Structure

```
caught-in-class/
│
├── faces/                     # Directory of known student images
├── CAUGHT_IN_CLASS.py         # Main application script
├── attendance_final_file.xlsx # Output file with attendance data
├── yolov8n.pt # YOLOv8 pre-trained model weights
├── images atendance.xlsx # Image-to-name mapping (optional depending on implementation)
├── requirements.txt # Requirements file
└── requirements.txt           # Required Python libraries
```

## Limitations & Future Enhancements

- Currently supports a single camera input.
- Assumes students face the camera directly for recognition.
- May add:
  - Multi-angle camera support
  - Attendance analytics dashboard
  - Integration with school databases or student portals

## Contributors

SPEED (mainly) and  **CAUGHT IN CLASS** team as part of a college semster project.
