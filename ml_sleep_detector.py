import cv2
import numpy as np
import pygame
import pickle
import threading
import time

class MLSleepDetector:
    def __init__(self):
        # Initialize pygame for sound
        pygame.mixer.init()
        
        # Detection parameters
        self.CLOSED_THRESHOLD = 0.6  # Confidence threshold for "eyes closed"
        self.CONSEC_FRAMES = 3  # Consecutive frames before alarm
        self.image_size = (64, 64)  # Same size used in training
        
        # Counters
        self.closed_counter = 0
        self.alarm_on = False
        
        # Load the trained model
        self.model = None
        self.load_model()
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        print("ML Sleep Detector Initialized!")
        print("Press 'q' to quit")
        print("System will beep when eyes are closed or no face detected")
    
    def load_model(self):
        """Load the trained eye classifier model"""
        try:
            with open('eye_classifier_model.pkl', 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.image_size = data['image_size']
            print("✅ Trained model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please run train_model.py first to create the model!")
            return False
    
    def preprocess_face(self, face_img):
        """Preprocess face image for model prediction"""
        try:
            # Convert to grayscale if needed
            if len(face_img.shape) == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Resize to model input size
            face_resized = cv2.resize(face_img, self.image_size)
            
            # Flatten and normalize
            face_flattened = face_resized.flatten() / 255.0
            
            return face_flattened
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None
    
    def predict_eye_state(self, face_img):
        """Predict if eyes are open or closed using the trained model"""
        if self.model is None:
            return "Unknown", 0.5
        
        try:
            # Preprocess the face image
            processed_face = self.preprocess_face(face_img)
            if processed_face is None:
                return "Error", 0.5
            
            # Make prediction
            prediction = self.model.predict([processed_face])[0]
            probabilities = self.model.predict_proba([processed_face])[0]
            
            # Get confidence
            confidence = max(probabilities)
            
            if prediction == 1:
                return "Eyes Open", confidence
            else:
                return "Eyes Closed", confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error", 0.5
    
    def play_beep(self):
        """Play beep sound"""
        def beep_thread():
            try:
                # Try to load beep.wav file
                beep_sound = pygame.mixer.Sound("beep.wav")
                beep_sound.play()
                
                # Wait for sound to finish
                while pygame.mixer.get_busy():
                    pygame.time.wait(10)
                    
            except pygame.error:
                # If beep.wav doesn't exist, create a simple beep
                print("BEEP! (beep.wav not found)")
            except Exception as e:
                print(f"Beep error: {e}")
        
        # Play beep in separate thread
        threading.Thread(target=beep_thread, daemon=True).start()
    
    def draw_text(self, frame, text, position, color=(0, 255, 0), font_scale=0.7):
        """Draw text on frame"""
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, 2, cv2.LINE_AA)
    
    def run(self):
        """Main detection loop"""
        if self.model is None:
            print("Cannot start detection without a trained model!")
            return
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    # Process the largest face
                    face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = face
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Extract face region with padding
                    padding = 20
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    
                    # Predict eye state
                    eye_state, confidence = self.predict_eye_state(face_crop)
                    
                    # Determine if alert is needed
                    if eye_state == "Eyes Closed" and confidence > self.CLOSED_THRESHOLD:
                        self.closed_counter += 1
                        
                        if self.closed_counter >= self.CONSEC_FRAMES:
                            if not self.alarm_on:
                                self.alarm_on = True
                                self.play_beep()
                                print("🚨 DROWSINESS ALERT!")
                            
                            # Draw alert
                            self.draw_text(frame, "DROWSINESS ALERT!", (10, 30), (0, 0, 255), 1.0)
                    else:
                        self.closed_counter = 0
                        self.alarm_on = False
                    
                    # Display information
                    color = (0, 255, 0) if eye_state == "Eyes Open" else (0, 0, 255)
                    self.draw_text(frame, f"{eye_state} ({confidence:.2%})", (10, 60), color)
                    self.draw_text(frame, f"Alert Counter: {self.closed_counter}", (10, 90))
                    
                else:
                    # No face detected - trigger alert
                    self.draw_text(frame, "NO FACE DETECTED - ALERT!", (10, 30), (0, 0, 255))
                    
                    if not self.alarm_on:
                        self.alarm_on = True
                        self.play_beep()
                        print("🚨 NO FACE DETECTED!")
                    
                    self.closed_counter = 0
                
                # Instructions
                self.draw_text(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                              (255, 255, 255), 0.6)
                
                # Show frame
                cv2.imshow('ML Sleep Detector', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nDetection stopped by user")
        except Exception as e:
            print(f"Detection error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        print("Sleep detector stopped.")

def main():
    try:
        detector = MLSleepDetector()
        detector.run()
    except Exception as e:
        print(f"Failed to start sleep detector: {e}")

if __name__ == "__main__":
    main()