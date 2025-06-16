import cv2
import os
import time

def save_screenshot(frame, gesture):
    """Save screenshot with timestamp"""
    os.makedirs("data/screenshots", exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"data/screenshots/{gesture}_{timestamp}.png"
    
    cv2.imwrite(filename, frame)
    return filename
