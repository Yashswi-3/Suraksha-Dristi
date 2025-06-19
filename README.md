# Hand Gesture & Face Recognition App

A simple Streamlit app for real-time hand gesture recognition and face detection using your webcam or an IP camera.

## Features

- Detects common hand gestures (Fist, Thumbs Up, Victory, Pointing, Open Hand)
- Detects faces when the "Victory" gesture is shown
- Automatically saves a screenshot when "Victory" is detected

## Requirements

- Python 3.7+
- streamlit
- opencv-python
- mediapipe
- numpy
- pillow

## Installation

pip install streamlit opencv-python mediapipe numpy pillow

## Usage

streamlit run app.py

- Select your camera type in the sidebar.
- Check the "Run" box to start the video stream.
- Show hand gestures in front of your camera.

## Output

- Screenshots are saved in the `img/` folder when the "Victory" gesture is detected.

---

**Enjoy using the app!**
