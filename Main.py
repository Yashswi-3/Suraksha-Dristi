import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import time
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to recognize gestures
def recognize_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]

    thumb_mcp = hand_landmarks.landmark[2]
    index_mcp = hand_landmarks.landmark[5]
    middle_mcp = hand_landmarks.landmark[9]
    ring_mcp = hand_landmarks.landmark[13]
    pinky_mcp = hand_landmarks.landmark[17]

    def is_finger_extended(tip, mcp, threshold=0.1):
        return tip.y < mcp.y - threshold

    thumb_extended = is_finger_extended(thumb_tip, thumb_mcp)
    index_extended = is_finger_extended(index_tip, index_mcp)
    middle_extended = is_finger_extended(middle_tip, middle_mcp)
    ring_extended = is_finger_extended(ring_tip, ring_mcp)
    pinky_extended = is_finger_extended(pinky_tip, pinky_mcp)

    fingers_extended = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
    extended_count = sum(fingers_extended)

    if extended_count == 0:
        return "Fist"
    if thumb_extended and not any(fingers_extended[1:]):
        return "Thumbs Up"
    if index_extended and middle_extended and not any([thumb_extended, ring_extended, pinky_extended]):
        return "Victory"
    if index_extended and not any([thumb_extended, middle_extended, ring_extended, pinky_extended]):
        return "Pointing"
    if all(fingers_extended):
        return "Open Hand"
    return "Unknown"

# Function to detect faces
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

# Streamlit App
st.title("🖐️ Hand Gesture & Face Recognition App")

st.sidebar.title("Camera Selection")
camera_type = st.sidebar.selectbox("Select Camera Type:", ["Laptop Webcam", "IP Camera"])

if camera_type == "Laptop Webcam":
    camera_index = 0
    ip_url = None
else:
    ip_url = st.sidebar.text_input("Enter IP Camera URL (e.g., http://192.168.0.101:8080/video):")
    camera_index = ip_url if ip_url else 0

run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(camera_index)
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture image")
                break

            # Flip the image for a later selfie-view display
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            gesture_detected = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    gesture = recognize_gesture(hand_landmarks)
                    cv2.putText(frame, gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

                    if gesture == "Victory":
                        gesture_detected = True

            if gesture_detected:
                # Save screenshot
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                directory = r"C:\Users\yashswi shukla\Desktop\Project\Suraksha_Dristi\img"
                screenshot_path = os.path.join(directory, f"screenshot_{timestamp}.png")

                # Make sure directory exists
                os.makedirs(directory, exist_ok=True)

                # Save the screenshot
                cv2.imwrite(screenshot_path, frame)
                st.success(f"Gesture detected! Screenshot saved as {screenshot_path}")


                # Detect faces
                faces = detect_faces(frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    face_img = frame[y:y+h, x:x+w]
                    st.image(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), caption="Detected Face")

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
