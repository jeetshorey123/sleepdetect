# Sleep Detector - Complete System Guide

## 🎯 Overview
This is a complete machine learning-based sleep detection system that:
1. Captures training photos of your eyes open and closed
2. Trains a machine learning model to classify eye states
3. Uses the trained model for accurate drowsiness detection
4. Beeps when eyes are closed or no face is detected

## 📁 Files Created
- `capture_photos.py` - Captures training photos
- `train_model.py` - Trains the ML model
- `ml_sleep_detector.py` - Main sleep detector with ML
- `sleep.py` - Original version (backup)

## 🚀 How to Use

### Step 1: Capture Training Photos
```bash
python capture_photos.py
```
- Press 'O' when your eyes are OPEN
- Press 'C' when your eyes are CLOSED
- Collect 50 photos each
- Press 'Q' to quit

### Step 2: Train the Model
```bash
python train_model.py
```
- Automatically trains on your photos
- Creates `eye_classifier_model.pkl`
- Shows accuracy results

### Step 3: Run Sleep Detector
```bash
python ml_sleep_detector.py
```
- Uses your trained model
- Beeps when drowsy or no face detected
- Press 'Q' to quit

## 📊 Features
✅ Manual photo collection for accurate training
✅ Machine learning classification (SVM)
✅ Real-time eye state prediction
✅ Beep alerts for drowsiness
✅ No face detection alerts
✅ Confidence scores displayed

## 📁 Directory Structure After Use
```
sleep detective/
├── capture_photos.py
├── train_model.py
├── ml_sleep_detector.py
├── sleep.py (original)
├── eye_classifier_model.pkl (trained model)
├── beep.wav (optional sound file)
└── dataset/
    ├── eyes_open/
    │   ├── open_001.jpg
    │   └── ... (50 photos)
    └── eyes_closed/
        ├── closed_001.jpg
        └── ... (50 photos)
```

## 🔧 Troubleshooting
- If accuracy is low (<70%), collect more diverse photos
- Ensure good lighting when capturing photos
- Make sure your face is clearly visible
- Add `beep.wav` file for custom alert sound