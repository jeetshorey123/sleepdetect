import cv2
import os
import time

def capture_images(label, save_dir, num_images=50, duration=20):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cap = cv2.VideoCapture(0)
    interval = duration / num_images
    print(f"Capturing {label} images. Please keep your eyes {label}.")
    for i in range(num_images):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            continue
        img_path = os.path.join(save_dir, f"{label}_{i+1}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        time.sleep(interval)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Eyes open
    capture_images('open', 'dataset/open')
    print("Waiting 10 seconds before capturing closed eyes images...")
    time.sleep(10)
    # Eyes closed
    capture_images('closed', 'dataset/closed')
    print("Image capture complete.")
