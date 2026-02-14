from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import pickle
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)

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

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        template_path = os.path.join(BASE_DIR, 'templates', 'index.html')
        with open(template_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        return f"Error loading page: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No JSON data provided'}), 400
        
        image_data = data.get('image')
        if not image_data:
            return jsonify({'status': 'error', 'message': 'No image data provided'}), 400
        
        result = detector.predict_eye_state(image_data)
        return jsonify(result)
        
    except Exception as e:
        print(f"Predict error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Alternative API endpoint"""
    return predict()

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Sleep Detective API is running',
        'model_loaded': detector.model is not None,
        'base_dir': BASE_DIR
    })

@app.route('/api/health')
def api_health():
    """Alternative health check endpoint"""
    return health()

# This is required for Vercel
def handler(event, context):
    """Serverless handler for Vercel"""
    return app(event, context)
