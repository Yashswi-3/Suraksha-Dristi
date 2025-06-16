import streamlit as st
import cv2
import mediapipe as mp
import time
import os
from collections import Counter

st.title("🖐️ Improved Gesture App")

# Initialize session state
if 'gesture_history' not in st.session_state:
    st.session_state.gesture_history = []
    st.session_state.stable_gesture = "Unknown"

def preprocess_frame(frame):
    """Improve frame quality for better detection"""
    alpha = 1.2  # Contrast
    beta = 10    # Brightness
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    return frame

def improved_gesture_recognition(landmarks):
    """More robust gesture recognition"""
    points = []
    for lm in landmarks.landmark:
        points.append([lm.x, lm.y])
    
    def is_finger_up(tip_idx, pip_idx, mcp_idx):
        tip_y = points[tip_idx][1]
        pip_y = points[pip_idx][1]
        mcp_y = points[mcp_idx][1]
        return tip_y < pip_y and tip_y < mcp_y
    
    fingers = []
    
    # Thumb
    if points[4][0] > points[3][0]:
        fingers.append(points[4][0] > points[2][0])
    else:
        fingers.append(points[4][0] < points[2][0])
    
    # Other fingers
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    finger_mcps = [5, 9, 13, 17]
    
    for tip, pip, mcp in zip(finger_tips, finger_pips, finger_mcps):
        fingers.append(is_finger_up(tip, pip, mcp))
    
    total_fingers = sum(fingers)
    
    if total_fingers == 0:
        return "Fist"
    elif total_fingers == 1:
        if fingers[0]:
            return "Thumbs Up"
        elif fingers[1]:
            return "Pointing"
        else:
            return "One"
    elif total_fingers == 2 and fingers[1] and fingers[2]:
        return "Victory"
    elif total_fingers == 5:
        return "High Five"
    else:
        return f"{total_fingers} Fingers"

def stabilize_gesture(current_gesture):
    """Smooth gesture detection"""
    st.session_state.gesture_history.append(current_gesture)
    
    if len(st.session_state.gesture_history) > 5:
        st.session_state.gesture_history.pop(0)
    
    if len(st.session_state.gesture_history) >= 3:
        most_common = Counter(st.session_state.gesture_history).most_common(1)[0][0]
        st.session_state.stable_gesture = most_common
    
    return st.session_state.stable_gesture

# Sidebar settings
st.sidebar.header("Detection Settings")
detection_confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.3)
tracking_confidence = st.sidebar.slider("Tracking Confidence", 0.1, 1.0, 0.3)

# Main app
run = st.checkbox("Start Camera")

if run:
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands.Hands(
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
        max_num_hands=2
    )
    mp_draw = mp.solutions.drawing_utils
    
    frame_placeholder = st.empty()
    status_placeholder = st.sidebar.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        frame = preprocess_frame(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb)
        
        current_gesture = "No Hand"
        
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                current_gesture = improved_gesture_recognition(hand)
        
        # Stabilize gesture
        stable_gesture = stabilize_gesture(current_gesture)
        
        # Display gesture
        cv2.putText(frame, stable_gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Update status
        status_placeholder.write(f"**Current Gesture:** {stable_gesture}")
        
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
