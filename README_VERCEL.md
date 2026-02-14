# Sleep Detective - Vercel Deployment

This is an AI-powered drowsiness detection system deployed on Vercel.

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (for desktop app)
python ml_sleep_detector.py

# Train the model
python train_model.py
```

## Vercel Deployment

The app is configured to run on Vercel as a serverless function.

### Structure:
- `/api/index.py` - Main serverless function
- `/api/test.py` - Test endpoint
- `/templates/index.html` - Web interface
- `/vercel.json` - Deployment configuration

### Endpoints:
- `/` - Main web interface
- `/predict` - Image analysis API
- `/health` - Health check
- `/api/test` - Simple test endpoint

## Technology Stack
- **Backend**: Python serverless functions
- **ML**: OpenCV, scikit-learn
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Deployment**: Vercel

## Repository
https://github.com/jeetshorey123/sleepdetect
