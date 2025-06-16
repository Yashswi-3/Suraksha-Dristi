def recognize_gesture(hand_landmarks):
    """Simple gesture recognition"""
    
    # Get landmark positions
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y])
    
    # Simple finger counting
    def finger_up(tip_idx, pip_idx):
        return landmarks[tip_idx][1] < landmarks[pip_idx][1]
    
    fingers = []
    # Thumb (special case)
    fingers.append(landmarks[4][0] > landmarks[3][0])
    # Other fingers
    fingers.extend([finger_up(8, 6), finger_up(12, 10), finger_up(16, 14), finger_up(20, 18)])
    
    total_fingers = sum(fingers)
    
    # Simple gesture mapping
    if total_fingers == 0:
        return "Fist"
    elif total_fingers == 1 and fingers[1]:  # Only index
        return "Pointing"
    elif total_fingers == 2 and fingers[1] and fingers[2]:  # Index + Middle
        return "Victory"
    elif total_fingers == 1 and fingers[0]:  # Only thumb
        return "Thumbs Up"
    elif total_fingers == 5:
        return "Open Hand"
    else:
        return "Unknown"

class GestureRecognizer:
    def recognize(self, hand_landmarks):
        return recognize_gesture(hand_landmarks)
