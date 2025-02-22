# Flood Area Human Detection System Using Raspberry Pi and Drone

This project leverages a Raspberry Pi mounted on a drone to detect humans in flood-affected areas. By combining TensorFlow Lite for real-time object detection and OpenCV for image processing, it provides an efficient solution for search and rescue missions.

---

### Features

- **Real-Time Human Detection**: Identifies humans in live video streams from a Pi Camera or USB Webcam.
- **Drone Integration**: Ensures wide-area surveillance during floods.
- **Customizable Settings**: Adjustable thresholds and resolution for accurate detection.
- **Edge TPU Support**: Optional support for Coral USB Accelerator to enhance performance.

---

### Getting Started

#### Hardware Requirements

- Raspberry Pi (with TensorFlow Lite support)
- Pi Camera or USB Webcam
- Drone for aerial deployment
- MicroSD card with Raspberry Pi OS installed
- Optional: Coral USB Accelerator for faster detection

#### Software Requirements

- Python 3.7+
- TensorFlow Lite / TFLite Runtime
- OpenCV
- NumPy
- Raspberry Pi OS

---

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/AlfinAkash/Human-Detection
   cd Human-Detection


2. **Install Dependencies**

Install the required Python packages by running the following command:


pip install -r requirements.txt


3. **Prepare Model and Labels**

Place your TensorFlow Lite model file (detect.tflite) inside the model/ directory.

Ensure the labelmap.txt file is included in the same directory.



4. **Run the Detection Script**

Run the following command to start the human detection:

python flood_human_detection.py --modeldir model --threshold 0.5 --resolution 1280x720



---

### Usage

**Arguments**

--modeldir: Path to the folder containing .tflite model and labelmap.txt.

--threshold: Minimum confidence threshold (default: 0.5).

--resolution: Video resolution (e.g., 1280x720)..


Example

python flood_human_detection.py --modeldir model --threshold 0.5 --resolution 640x480


---

### Testing Results 

![Pic1](https://github.com/AlfinAkash/Human-Detection/blob/4c43639e39768f70053f359fdfb73b38565ab475/pic1.jpg)

![Pic2](https://github.com/AlfinAkash/Human-Detection/blob/afaef1e0a93a23c4da7d41d3174545711a8c1e47/pic2.jpg)

---

### Reference Article 

[Human Object Detection Documentation](https://github.com/AlfinAkash/Human-Detection/blob/7e9fa02d1d2f118a8b0e7cf9e7cc3ec411ee9e59/Human_Object_Detection_for_Real-Time_Camera_using_.pdf)

---

### Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your updates or improvements.


---

### Contributors

- [Akash A](https://github.com/AlfinAkash)  
- [Hariharan L](https://github.com/Hariharan)  
- [Karthickvikraman M](https://github.com/karthickvikraman22)




