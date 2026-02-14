import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

def preprocess_image(img):
    # Resize and normalize image for model input
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_eye_state(model, frame):
    img = preprocess_image(frame)
    pred = model.predict(img)
    return 'open' if pred[0][0] > 0.5 else 'closed'

if __name__ == "__main__":
    # Load your trained model (update path as needed)
    model_path = 'eye_state_model.h5'  # Replace with your model path
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        exit(1)
    model = load_model(model_path)
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        state = predict_eye_state(model, frame)
        cv2.putText(frame, f"Eyes: {state}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Eye State Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
