import cv2
import os
import time
import numpy as np

class PhotoCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # Create directories
        self.create_directories()
        
        # Counters
        self.eyes_open_count = 0
        self.eyes_closed_count = 0
        self.max_photos = 50
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        print("Photo Capture Initialized!")
        print("Instructions:")
        print("- Press 'o' to capture photo with EYES OPEN")
        print("- Press 'c' to capture photo with EYES CLOSED")
        print("- Press 'q' to quit")
        print(f"Target: {self.max_photos} photos each of eyes open and closed")
    
    def create_directories(self):
        """Create directories for storing photos"""
        os.makedirs('dataset/eyes_open', exist_ok=True)
        os.makedirs('dataset/eyes_closed', exist_ok=True)
        print("Created directories: dataset/eyes_open and dataset/eyes_closed")
    
    def save_face_crop(self, frame, label):
        """Save cropped face image"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Get the largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face
            
            # Crop face region with some padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if label == 'open' and self.eyes_open_count < self.max_photos:
                filename = f"dataset/eyes_open/open_{self.eyes_open_count + 1:03d}.jpg"
                cv2.imwrite(filename, face_crop)
                self.eyes_open_count += 1
                print(f"Saved {filename} ({self.eyes_open_count}/{self.max_photos} eyes open)")
                return True
            elif label == 'closed' and self.eyes_closed_count < self.max_photos:
                filename = f"dataset/eyes_closed/closed_{self.eyes_closed_count + 1:03d}.jpg"
                cv2.imwrite(filename, face_crop)
                self.eyes_closed_count += 1
                print(f"Saved {filename} ({self.eyes_closed_count}/{self.max_photos} eyes closed)")
                return True
        else:
            print("No face detected! Make sure your face is visible.")
        
        return False
    
    def run(self):
        """Main capture loop"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces and draw rectangles
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Detect eyes within face
                    face_gray = gray[y:y+h, x:x+w]
                    eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 4)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
                
                # Display instructions and progress
                cv2.putText(frame, f"Eyes Open: {self.eyes_open_count}/{self.max_photos}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Eyes Closed: {self.eyes_closed_count}/{self.max_photos}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Press 'O' for Open, 'C' for Closed, 'Q' to Quit", 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Check if collection is complete
                if self.eyes_open_count >= self.max_photos and self.eyes_closed_count >= self.max_photos:
                    cv2.putText(frame, "COLLECTION COMPLETE! Press 'Q' to exit", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                cv2.imshow('Photo Capture', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('o') or key == ord('O'):
                    if self.eyes_open_count < self.max_photos:
                        self.save_face_crop(frame, 'open')
                    else:
                        print("Already have enough 'eyes open' photos!")
                elif key == ord('c') or key == ord('C'):
                    if self.eyes_closed_count < self.max_photos:
                        self.save_face_crop(frame, 'closed')
                    else:
                        print("Already have enough 'eyes closed' photos!")
        
        except KeyboardInterrupt:
            print("\nCapture stopped by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\nCapture complete!")
        print(f"Eyes open photos: {self.eyes_open_count}")
        print(f"Eyes closed photos: {self.eyes_closed_count}")

if __name__ == "__main__":
    try:
        capturer = PhotoCapture()
        capturer.run()
    except Exception as e:
        print(f"Error: {e}")