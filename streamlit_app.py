import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Sleep Detective",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-size: 1.1rem;
        border-radius: 8px;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .eyes-open {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .eyes-closed {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

class SleepDetectorApp:
    def __init__(self):
        self.model = None
        self.image_size = (64, 64)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def load_model(self):
        """Load the trained eye classifier model"""
        try:
            with open('eye_classifier_model.pkl', 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.image_size = data['image_size']
            return True
        except Exception as e:
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
            st.error(f"Error preprocessing image: {e}")
            return None
    
    def predict_eye_state(self, image):
        """Predict if eyes are open or closed"""
        if self.model is None:
            return None, 0, None
        
        try:
            # Convert PIL image to OpenCV format
            img = np.array(image)
            if len(img.shape) == 2:  # Grayscale
                gray = img
            else:  # Color
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return "No Face Detected", 0, None
            
            # Get the first face
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Preprocess and predict
            processed_face = self.preprocess_face(face_roi)
            if processed_face is None:
                return "Error", 0, None
            
            # Make prediction
            prediction = self.model.predict([processed_face])[0]
            probabilities = self.model.predict_proba([processed_face])[0]
            confidence = max(probabilities)
            
            # Draw rectangle on face
            result_img = img.copy()
            color = (0, 255, 0) if prediction == 1 else (255, 0, 0)
            cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
            
            state = "Eyes Open" if prediction == 1 else "Eyes Closed"
            
            return state, confidence, result_img
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return "Error", 0, None

def main():
    # Header
    st.title("😴 Sleep Detective")
    st.markdown("### AI-Powered Drowsiness Detection System")
    
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = SleepDetectorApp()
    
    detector = st.session_state.detector
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        **Sleep Detective** uses machine learning to detect drowsiness by analyzing eye states.
        
        **How to use:**
        1. Upload a clear photo of your face
        2. The AI will detect your eyes
        3. Get instant drowsiness prediction
        """)
        
        st.header("Model Status")
        if detector.load_model():
            st.success("✅ Model Loaded")
        else:
            st.warning("⚠️ No trained model found. Using demo mode.")
            st.info("To train your own model, clone the repository and run `train_model.py`")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a photo of your face",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image showing your face"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.header("Detection Result")
        
        if uploaded_file is not None:
            with st.spinner("Analyzing..."):
                state, confidence, result_img = detector.predict_eye_state(image)
                
                if state == "No Face Detected":
                    st.warning("⚠️ No face detected in the image. Please upload a clearer photo.")
                elif state == "Error":
                    st.error("❌ Error processing image")
                elif state:
                    # Show result image with face detection box
                    if result_img is not None:
                        st.image(result_img, caption="Processed Image", use_column_width=True)
                    
                    # Show prediction
                    css_class = "eyes-open" if state == "Eyes Open" else "eyes-closed"
                    st.markdown(f"""
                        <div class="prediction-box {css_class}">
                            {state}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Show confidence
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                    
                    # Show warning if eyes closed
                    if state == "Eyes Closed":
                        st.warning("⚠️ **ALERT:** Eyes appear to be closed! This indicates drowsiness.")
        else:
            st.info("👆 Upload an image to get started")
    
    # Instructions
    st.markdown("---")
    st.header("📖 Instructions")
    
    tab1, tab2, tab3 = st.tabs(["Upload & Detect", "System Info", "GitHub"])
    
    with tab1:
        st.markdown("""
        ### How to Use Sleep Detective
        
        1. **Take or upload a photo** showing your face clearly
        2. **Upload the image** using the file uploader
        3. **Wait for analysis** - the AI will detect your face and eye state
        4. **View results** - see if your eyes are open or closed
        
        **Tips for best results:**
        - Ensure good lighting
        - Face the camera directly
        - Keep your face clearly visible
        - Avoid wearing sunglasses
        """)
    
    with tab2:
        st.markdown("""
        ### System Information
        
        **Technology Stack:**
        - **Frontend:** Streamlit
        - **ML Model:** Support Vector Machine (SVM)
        - **Face Detection:** OpenCV Haar Cascades
        - **Image Processing:** OpenCV & NumPy
        
        **Features:**
        - Real-time eye state classification
        - Face detection and tracking
        - Confidence scoring
        - Visual feedback with bounding boxes
        """)
    
    with tab3:
        st.markdown("""
        ### GitHub Repository
        
        🔗 [github.com/jeetshorey123/sleepdetect](https://github.com/jeetshorey123/sleepdetect)
        
        **To run locally:**
        ```bash
        # Clone the repository
        git clone https://github.com/jeetshorey123/sleepdetect.git
        cd sleepdetect
        
        # Install dependencies
        pip install -r requirements.txt
        
        # Capture training photos
        python capture_photos.py
        
        # Train the model
        python train_model.py
        
        # Run the detector
        python ml_sleep_detector.py
        ```
        """)

if __name__ == "__main__":
    main()
