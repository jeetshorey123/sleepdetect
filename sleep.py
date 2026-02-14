import cv2
import numpy as np
import pygame
import threading
import time
from scipy.spatial import distance as dist

class SleepDetector:
    def __init__(self):
        # Initialize pygame for sound
        pygame.mixer.init()
        
        # Eye aspect ratio threshold and consecutive frame parameters
        self.EYE_AR_THRESH = 0.25  # Threshold for detecting closed eyes
        self.EYE_AR_CONSEC_FRAMES = 3  # Number of consecutive frames eye must be closed (about 0.1-0.3 seconds)
        
        # Initialize counters
        self.COUNTER = 0
        self.ALARM_ON = False
        
        # Photo capture counters
        self.photos_eyes_closed = 0
        self.photos_eyes_open = 0
        self.max_photos = 50
        self.last_photo_time = 0
        self.photo_delay = 1.0  # Minimum delay between photos (seconds)
        
        # Load face and eye cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        print("Sleep Detector Initialized!")
        print("Press 'q' to quit the application")
        print("If your eyes are closed for too long, an alarm will sound!")
        print(f"Will capture {self.max_photos} photos with eyes closed and {self.max_photos} with eyes open")
        print("Photos will be saved as 'eyes_closed_X.jpg' and 'eyes_open_X.jpg'")
        
        # Create directories for photos
        import os
        if not os.path.exists('photos_eyes_closed'):
            os.makedirs('photos_eyes_closed')
        if not os.path.exists('photos_eyes_open'):
            os.makedirs('photos_eyes_open')
    
    def eye_aspect_ratio(self, eye_points):
        """Calculate the eye aspect ratio (EAR) for eye landmarks"""
        # Convert to numpy array if needed
        if len(eye_points) < 6:
            return 0.3  # Default value if not enough points
        
        # Calculate vertical eye distances
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        
        # Calculate horizontal eye distance
        C = dist.euclidean(eye_points[0], eye_points[3])
        
        # Calculate the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_eyes_simple(self, gray_face):
        """Simple eye detection using eye cascade"""
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 4)
        
        if len(eyes) >= 2:
            # Calculate simple EAR based on eye rectangle dimensions
            total_ear = 0
            for (ex, ey, ew, eh) in eyes[:2]:  # Take first two eyes
                # Simple approximation: height to width ratio
                ear = eh / ew if ew > 0 else 0.3
                total_ear += ear
            
            avg_ear = total_ear / len(eyes[:2])
            return avg_ear
        
        return 0.3  # Default value when no eyes detected
    
    def save_photo(self, frame, eyes_closed=True):
        """Save photo for training data"""
        current_time = time.time()
        if current_time - self.last_photo_time < self.photo_delay:
            return False
        
        if eyes_closed and self.photos_eyes_closed < self.max_photos:
            filename = f"photos_eyes_closed/eyes_closed_{self.photos_eyes_closed + 1:03d}.jpg"
            cv2.imwrite(filename, frame)
            self.photos_eyes_closed += 1
            self.last_photo_time = current_time
            print(f"Saved {filename} ({self.photos_eyes_closed}/{self.max_photos} eyes closed)")
            return True
        elif not eyes_closed and self.photos_eyes_open < self.max_photos:
            filename = f"photos_eyes_open/eyes_open_{self.photos_eyes_open + 1:03d}.jpg"
            cv2.imwrite(filename, frame)
            self.photos_eyes_open += 1
            self.last_photo_time = current_time
            print(f"Saved {filename} ({self.photos_eyes_open}/{self.max_photos} eyes open)")
            return True
        
        return False
        """Play beep sound from beep.wav file"""
        def play_beep():
            if not self.ALARM_ON:
                return
            
            try:
                # Load and play the beep.wav file
                beep_sound = pygame.mixer.Sound("beep.wav")
                beep_sound.play()
                
                # Wait for the sound to finish playing
                while pygame.mixer.get_busy():
                    pygame.time.wait(10)
                    
            except pygame.error as e:
                print(f"Could not load beep.wav: {e}")
                print("Make sure beep.wav file exists in the same directory as sleep.py")
            except Exception as e:
                print(f"Error playing beep sound: {e}")
        
        # Run beep in separate thread to avoid blocking
        beep_thread = threading.Thread(target=play_beep)
        beep_thread.daemon = True
        beep_thread.start()
    
    def draw_text(self, frame, text, position, color=(0, 255, 0), font_scale=0.7):
        """Helper function to draw text on frame"""
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, 2)
    
    def run(self):
        """Main detection loop"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    # Process the first detected face
                    (x, y, w, h) = faces[0]
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Extract face region
                    gray_face = gray[y:y+h, x:x+w]
                    
                    # Detect eyes and calculate EAR
                    ear = self.detect_eyes_simple(gray_face)
                    
                    # Check if eyes are closed
                    if ear < self.EYE_AR_THRESH:
                        self.COUNTER += 1
                        
                        # If eyes have been closed for sufficient consecutive frames
                        if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                            if not self.ALARM_ON:
                                self.ALARM_ON = True
                                print("DROWSINESS ALERT!")
                                
                            # Sound the alarm
                            self.create_beep_sound()
                            
                            # Draw alert text
                            self.draw_text(frame, "DROWSINESS ALERT!", (10, 30), (0, 0, 255), 1.0)
                            
                            # Save photo of eyes closed
                            self.save_photo(frame, eyes_closed=True)
                    else:
                        self.COUNTER = 0
                        self.ALARM_ON = False
                        
                        # Save photo of eyes open
                        self.save_photo(frame, eyes_closed=False)
                    
                    # Display EAR and status
                    self.draw_text(frame, f"EAR: {ear:.2f}", (10, 60))
                    self.draw_text(frame, f"Counter: {self.COUNTER}", (10, 90))
                    
                    # Draw eyes detection
                    eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 4)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
                
                else:
                    # No face detected - BEEP!
                    self.draw_text(frame, "No face detected - ALERT!", (10, 30), (0, 0, 255))
                    self.COUNTER = 0
                    if not self.ALARM_ON:
                        self.ALARM_ON = True
                        self.create_beep_sound()  # Beep when no face detected
                    else:
                        self.ALARM_ON = False  # Reset for next detection
                
                # Display photo capture progress
                self.draw_text(frame, f"Photos: Closed {self.photos_eyes_closed}/{self.max_photos}, Open {self.photos_eyes_open}/{self.max_photos}", 
                              (10, 120), (255, 255, 0), 0.6)
                
                # Display instructions
                self.draw_text(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), (255, 255, 255), 0.5)
                
                # Show the frame
                cv2.imshow('Sleep Detector', frame)
                
                # Check for quit command
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nDetection stopped by user")
        
        except Exception as e:
            print(f"Error during detection: {e}")
        
        finally:
            # Clean up
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        print("Sleep detector stopped.")

def main():
    """Main function to run the sleep detector"""
    try:
        # Create and run sleep detector
        detector = SleepDetector()
        detector.run()
        
    except Exception as e:
        print(f"Failed to initialize sleep detector: {e}")
        print("Make sure your camera is connected and not being used by another application.")

if __name__ == "__main__":
    main()
