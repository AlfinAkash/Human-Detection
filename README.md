
---

Flood Area Human Detection System Using Raspberry Pi and Drone

This project leverages a Raspberry Pi mounted on a drone to detect humans in flood-affected areas. By combining TensorFlow Lite for real-time object detection and OpenCV for image processing, it provides an efficient solution for search and rescue missions.


---

Features

Real-Time Human Detection: Identifies humans in live video streams from a Pi Camera or USB Webcam.

Drone Integration: Ensures wide-area surveillance during floods.

Customizable Settings: Adjustable thresholds and resolution for accurate detection.

Edge TPU Support: Optional support for Coral USB Accelerator to enhance performance.



---

Getting Started

Hardware Requirements

Raspberry Pi (with TensorFlow Lite support)

Pi Camera or USB Webcam

Drone for aerial deployment

MicroSD card with Raspberry Pi OS installed

Optional: Coral USB Accelerator for faster detection


Software Requirements

Python 3.7+

TensorFlow Lite / TFLite Runtime

OpenCV

NumPy

Raspberry Pi OS



---

Installation

1. Clone the Repository

git clone https://github.com/AlfinAkash/Human Detection.git
cd Human Detection 

2. Install Dependencies

pip install -r requirements.txt

3. Prepare Model and Labels

Place your TensorFlow Lite model file (detect.tflite) inside the model/ directory.

Include the corresponding labelmap.txt file in the same directory.


4. Run the Detection Script

python flood_human_detection.py --modeldir model --threshold 0.5 --resolution 1280x720


---

Code Overview

Main Detection Script

from threading import Thread
import cv2
import numpy as np
import os
import time
from tflite_runtime.interpreter import Interpreter

class VideoStream:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])
        self.stopped = False
        self.grabbed, self.frame = self.stream.read()

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# Define detection settings
model_path = "model/detect.tflite"
labels_path = "model/labelmap.txt"

# Load model and labels
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
labels = [line.strip() for line in open(labels_path, 'r').readlines()]

# Initialize video stream
videostream = VideoStream(resolution=(640, 480), framerate=30).start()
time.sleep(1)

while True:
    frame = videostream.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(cv2.resize(frame_rgb, (300, 300)), axis=0)

    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()

    # Get detection results
    boxes = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
    classes = interpreter.get_tensor(interpreter.get_output_details()[1]['index'])[0]
    scores = interpreter.get_tensor(interpreter.get_output_details()[2]['index'])[0]

    for i in range(len(scores)):
        if scores[i] > 0.5:  # Threshold for detection
            ymin = int(boxes[i][0] * frame.shape[0])
            xmin = int(boxes[i][1] * frame.shape[1])
            ymax = int(boxes[i][2] * frame.shape[0])
            xmax = int(boxes[i][3] * frame.shape[1])

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"{labels[int(classes[i])]}: {int(scores[i] * 100)}%"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Flood Human Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

videostream.stop()
cv2.destroyAllWindows()


---

Usage

Arguments

--modeldir: Path to the folder containing .tflite model and labelmap.txt.

--threshold: Minimum confidence threshold (default: 0.5).

--resolution: Video resolution (e.g., 1280x720).


Example:

python flood_human_detection.py --modeldir model --threshold 0.5 --resolution 640x480


---



---

Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your updates or improvements.


---

License

This project is licensed under the MIT License. See the LICENSE file for details.


---

Let me know if you'd like to include sample outputs or additional sections!

