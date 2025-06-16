import cv2

class CameraManager:
    def __init__(self):
        self.cap = None
    
    def start(self, source=0):
        self.cap = cv2.VideoCapture(source)
        return self.cap.isOpened()
    
    def read(self):
        if self.cap:
            return self.cap.read()
        return False, None
    
    def stop(self):
        if self.cap:
            self.cap.release()
