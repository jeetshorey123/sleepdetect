from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pickle
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__, template_folder='../templates', static_folder='../static')

class SleepDetector:
    def __init__(self):
        self.model = None
        self.image_size = (64, 64)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.load_model()
        
    def load_model(self):
        """Load the trained eye classifier model"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'eye_classifier_model.pkl')
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.image_size = data['image_size']
            return True
        except Exception as e:
            print(f"Model not loaded: {e}")
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
            img_bytes = base64.b64decode(image_data.split(',')[1])
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
                    'message': 'No face detected',
                    'state': 'No Face Detected',
                    'confidence': 0
                }
            
            # Get the first face
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Draw rectangle on face
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            
            # Preprocess and predict
            if self.model is None:
                return {
                    'status': 'error',
                    'message': 'Model not loaded. Using demo mode.',
                    'state': 'Unknown',
                    'confidence': 0
                }
            
            processed_face = self.preprocess_face(face_roi)
            if processed_face is None:
                return {
                    'status': 'error',
                    'message': 'Error processing face',
                    'state': 'Error',
                    'confidence': 0
                }
            
            # Make prediction
            prediction = self.model.predict([processed_face])[0]
            probabilities = self.model.predict_proba([processed_face])[0]
            confidence = float(max(probabilities))
            
            state = "Eyes Open" if prediction == 1 else "Eyes Closed"
            
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
            return {
                'status': 'error',
                'message': str(e),
                'state': 'Error',
                'confidence': 0
            }

# Initialize detector
detector = SleepDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'status': 'error', 'message': 'No image provided'}), 400
        
        result = detector.predict_eye_state(image_data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': detector.model is not None
    })

# For Vercel
def handler(request):
    return app(request)
