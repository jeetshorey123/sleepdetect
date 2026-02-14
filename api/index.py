from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
import cv2
import numpy as np
import pickle
import base64
from io import BytesIO
from PIL import Image
import os
import sys

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class SleepDetector:
    def __init__(self):
        self.model = None
        self.image_size = (64, 64)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.load_model()
        
    def load_model(self):
        """Load the trained eye classifier model"""
        try:
            model_path = os.path.join(BASE_DIR, 'eye_classifier_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.image_size = data['image_size']
                print("✅ Model loaded successfully", file=sys.stderr)
                return True
            else:
                print(f"⚠️ Model file not found at: {model_path}", file=sys.stderr)
                return False
        except Exception as e:
            print(f"❌ Model not loaded: {e}", file=sys.stderr)
            return False
    
    def preprocess_face(self, face_img):
        """Preprocess face image for model prediction"""
        try:
            if len(face_img.shape) == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            face_resized = cv2.resize(face_img, self.image_size)
            face_flattened = face_resized.flatten() / 255.0
            
            return face_flattened
        except Exception as e:
            print(f"Error preprocessing: {e}", file=sys.stderr)
            return None
    
    def predict_eye_state(self, image_data):
        """Predict if eyes are open or closed"""
        try:
            # Decode base64 image
            if ',' in image_data:
                img_bytes = base64.b64decode(image_data.split(',')[1])
            else:
                img_bytes = base64.b64decode(image_data)
            
            img = Image.open(BytesIO(img_bytes))
            img = np.array(img)
            
            # Convert to BGR for OpenCV
            if len(img.shape) == 2:
                gray = img
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return {
                    'status': 'error',
                    'message': 'No face detected in the image. Please ensure your face is clearly visible.',
                    'state': 'No Face Detected',
                    'confidence': 0
                }
            
            # Get the first face
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Draw rectangle on face
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
            
            # Preprocess and predict
            if self.model is None:
                return {
                    'status': 'error',
                    'message': 'Model not loaded. Demo mode - upload a clear face image.',
                    'state': 'Model Not Available',
                    'confidence': 0
                }
            
            processed_face = self.preprocess_face(face_roi)
            if processed_face is None:
                return {
                    'status': 'error',
                    'message': 'Error processing face region',
                    'state': 'Error',
                    'confidence': 0
                }
            
            # Make prediction
            prediction = self.model.predict([processed_face])[0]
            probabilities = self.model.predict_proba([processed_face])[0]
            confidence = float(max(probabilities))
            
            state = "Eyes Open" if prediction == 1 else "Eyes Closed"
            
            # Update color based on prediction
            color = (0, 255, 0) if prediction == 1 else (255, 0, 0)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
            
            # Add text label
            label = f"{state} ({confidence*100:.1f}%)"
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Encode result image
            _, buffer = cv2.imencode('.jpg', img)
            result_image = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'status': 'success',
                'state': state,
                'confidence': confidence,
                'result_image': f'data:image/jpeg;base64,{result_image}',
                'alert': state == "Eyes Closed"
            }
            
        except Exception as e:
            print(f"Prediction error: {e}", file=sys.stderr)
            return {
                'status': 'error',
                'message': f'Error processing image: {str(e)}',
                'state': 'Error',
                'confidence': 0
            }

# Initialize detector globally
detector = SleepDetector()

class handler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        try:
            if path == '/' or path == '/index.html':
                # Serve the main HTML page
                template_path = os.path.join(BASE_DIR, 'templates', 'index.html')
                if os.path.exists(template_path):
                    with open(template_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(html_content.encode('utf-8'))
                else:
                    self.send_error(404, 'Template not found')
                    
            elif path == '/health' or path == '/api/health':
                # Health check endpoint
                response = {
                    'status': 'ok',
                    'message': 'Sleep Detective API is running',
                    'model_loaded': detector.model is not None,
                    'base_dir': BASE_DIR
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
            else:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_response = {'error': 'Not found', 'path': path}
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
                
        except Exception as e:
            print(f"GET error: {e}", file=sys.stderr)
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        try:
            if path == '/predict' or path == '/api/predict':
                # Get request body
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                
                # Parse JSON
                data = json.loads(post_data.decode('utf-8'))
                image_data = data.get('image')
                
                if not image_data:
                    response = {'status': 'error', 'message': 'No image data provided'}
                    self.send_response(400)
                else:
                    # Predict
                    response = detector.predict_eye_state(image_data)
                    self.send_response(200)
                
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
            else:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Not found'}).encode('utf-8'))
                
        except Exception as e:
            print(f"POST error: {e}", file=sys.stderr)
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

class SleepDetector:
    def __init__(self):
        self.model = None
        self.image_size = (64, 64)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.load_model()
        
    def load_model(self):
        """Load the trained eye classifier model"""
        try:
            model_path = os.path.join(BASE_DIR, 'eye_classifier_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.image_size = data['image_size']
                print("✅ Model loaded successfully")
                return True
            else:
                print(f"⚠️ Model file not found at: {model_path}")
                return False
        except Exception as e:
            print(f"❌ Model not loaded: {e}")
            return False
    
    def preprocess_face(self, face_img):
        """Preprocess face image for model prediction"""
        try:
            if len(face_img.shape) == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            face_resized = cv2.resize(face_img, self.image_size)
            face_flattened = face_resized.flatten() / 255.0
            
            return face_flattened
        except Exception as e:
            print(f"Error preprocessing: {e}")
            return None
    
    def predict_eye_state(self, image_data):
        """Predict if eyes are open or closed"""
        try:
            # Decode base64 image
            if ',' in image_data:
                img_bytes = base64.b64decode(image_data.split(',')[1])
            else:
                img_bytes = base64.b64decode(image_data)
            
            img = Image.open(BytesIO(img_bytes))
            img = np.array(img)
            
            # Convert to BGR for OpenCV
            if len(img.shape) == 2:
                gray = img
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return {
                    'status': 'error',
                    'message': 'No face detected in the image. Please ensure your face is clearly visible.',
                    'state': 'No Face Detected',
                    'confidence': 0
                }
            
            # Get the first face
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Draw rectangle on face
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
            
            # Preprocess and predict
            if self.model is None:
                return {
                    'status': 'error',
                    'message': 'Model not loaded. Demo mode - upload a clear face image.',
                    'state': 'Model Not Available',
                    'confidence': 0
                }
            
            processed_face = self.preprocess_face(face_roi)
            if processed_face is None:
                return {
                    'status': 'error',
                    'message': 'Error processing face region',
                    'state': 'Error',
                    'confidence': 0
                }
            
            # Make prediction
            prediction = self.model.predict([processed_face])[0]
            probabilities = self.model.predict_proba([processed_face])[0]
            confidence = float(max(probabilities))
            
            state = "Eyes Open" if prediction == 1 else "Eyes Closed"
            
            # Update color based on prediction
            color = (0, 255, 0) if prediction == 1 else (255, 0, 0)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
            
            # Add text label
            label = f"{state} ({confidence*100:.1f}%)"
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Encode result image
            _, buffer = cv2.imencode('.jpg', img)
            result_image = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'status': 'success',
                'state': state,
                'confidence': confidence,
                'result_image': f'data:image/jpeg;base64,{result_image}',
                'alert': state == "Eyes Closed"
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'status': 'error',
                'message': f'Error processing image: {str(e)}',
                'state': 'Error',
                'confidence': 0
            }

# Initialize detector
detector = SleepDetector()

class handler(BaseHTTPRequestHandler):
    """Serverless handler for Vercel"""
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path == '/' or self.path == '/index.html':
                # Serve the main HTML page
                template_path = os.path.join(BASE_DIR, 'templates', 'index.html')
                with open(template_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(html_content.encode('utf-8'))
                
            elif self.path == '/health' or self.path == '/api/health':
                # Health check endpoint
                response = {
                    'status': 'ok',
                    'message': 'Sleep Detective API is running',
                    'model_loaded': detector.model is not None
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
            else:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Not found'}).encode('utf-8'))
                
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            if self.path == '/predict' or self.path == '/api/predict':
                # Get request body
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                
                # Parse JSON
                data = json.loads(post_data.decode('utf-8'))
                image_data = data.get('image')
                
                if not image_data:
                    response = {'status': 'error', 'message': 'No image data provided'}
                    self.send_response(400)
                else:
                    # Predict
                    response = detector.predict_eye_state(image_data)
                    self.send_response(200)
                
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
            else:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Not found'}).encode('utf-8'))
                
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
