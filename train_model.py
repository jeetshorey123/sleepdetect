import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
from PIL import Image

class EyeClassifierTrainer:
    def __init__(self):
        self.model = None
        self.image_size = (64, 64)  # Resize all images to this size
        
    def load_images(self, folder_path, label):
        """Load all images from a folder and assign them a label"""
        images = []
        labels = []
        
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist!")
            return images, labels
        
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in image_files:
            try:
                img_path = os.path.join(folder_path, filename)
                # Load image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Resize image
                    img = cv2.resize(img, self.image_size)
                    # Flatten the image to create feature vector
                    img_flattened = img.flatten()
                    # Normalize pixel values
                    img_normalized = img_flattened / 255.0
                    
                    images.append(img_normalized)
                    labels.append(label)
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        return images, labels
    
    def prepare_dataset(self):
        """Load and prepare the complete dataset"""
        print("Loading dataset...")
        
        # Load eyes open images (label = 1)
        open_images, open_labels = self.load_images('dataset/eyes_open', 1)
        print(f"Loaded {len(open_images)} eyes open images")
        
        # Load eyes closed images (label = 0)
        closed_images, closed_labels = self.load_images('dataset/eyes_closed', 0)
        print(f"Loaded {len(closed_images)} eyes closed images")
        
        if len(open_images) == 0 or len(closed_images) == 0:
            raise Exception("No images found! Please run capture_photos.py first to collect training data.")
        
        # Combine all images and labels
        all_images = open_images + closed_images
        all_labels = open_labels + closed_labels
        
        # Convert to numpy arrays
        X = np.array(all_images)
        y = np.array(all_labels)
        
        print(f"Total dataset: {len(X)} images")
        print(f"Features per image: {X.shape[1]}")
        print(f"Eyes open samples: {np.sum(y == 1)}")
        print(f"Eyes closed samples: {np.sum(y == 0)}")
        
        return X, y
    
    def train_model(self):
        """Train the SVM classifier"""
        print("Preparing dataset...")
        X, y = self.prepare_dataset()
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train SVM classifier
        print("Training SVM classifier...")
        self.model = SVC(kernel='rbf', probability=True, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.2%}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Eyes Closed', 'Eyes Open']))
        
        return accuracy
    
    def save_model(self, filename='eye_classifier_model.pkl'):
        """Save the trained model to a file"""
        if self.model is None:
            print("No model to save! Train the model first.")
            return
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'image_size': self.image_size
                }, f)
            print(f"Model saved as {filename}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filename='eye_classifier_model.pkl'):
        """Load a trained model from file"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.image_size = data['image_size']
            print(f"Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_single_image(self, image_path):
        """Predict if eyes are open or closed for a single image"""
        if self.model is None:
            print("No model loaded! Train or load a model first.")
            return None
        
        try:
            # Load and preprocess image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.image_size)
            img_flattened = img.flatten() / 255.0
            
            # Make prediction
            prediction = self.model.predict([img_flattened])[0]
            probability = self.model.predict_proba([img_flattened])[0]
            
            result = "Eyes Open" if prediction == 1 else "Eyes Closed"
            confidence = max(probability)
            
            return result, confidence
        
        except Exception as e:
            print(f"Error predicting image: {e}")
            return None

def main():
    trainer = EyeClassifierTrainer()
    
    try:
        # Train the model
        accuracy = trainer.train_model()
        
        # Save the model if accuracy is decent
        if accuracy > 0.7:  # 70% accuracy threshold
            trainer.save_model()
            print(f"\n✅ Model training successful! Accuracy: {accuracy:.2%}")
            print("Model saved as 'eye_classifier_model.pkl'")
        else:
            print(f"\n⚠️ Model accuracy ({accuracy:.2%}) is low. Consider collecting more/better training data.")
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("Make sure you have collected training data using capture_photos.py first!")

if __name__ == "__main__":
    main()