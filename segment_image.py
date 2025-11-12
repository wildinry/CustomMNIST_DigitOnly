import cv2
import numpy as np
import os
import shutil

INPUT_IMAGE = 'custom_hand_image.jpg'
OUTPUT_ROOT_DIR = 'custom_dataset/unlabeled'
TARGET_SIZE = (28, 28)

# Warning: This deletes the previous unlabeled folder and its contents
if os.path.exists(OUTPUT_ROOT_DIR):
    shutil.rmtree(OUTPUT_ROOT_DIR) 
os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
print(f"Output directory created at: {OUTPUT_ROOT_DIR}")

def segment_and_save_digits():
    """Loads the grid image, finds contours, and saves individual digits."""
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image file '{INPUT_IMAGE}'. Ensure it's in the same directory.")
        return

    inverted_img = cv2.bitwise_not(img) 

    _, thresh = cv2.threshold(inverted_img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_crops = []
    min_area = 500 
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = w / h
        
        if area > min_area and w > 15 and h > 15 and 0.2 < aspect_ratio < 2.0:
            
            padding = 10
            y_start = max(0, y - padding)
            y_end = min(img.shape[0], y + h + padding)
            x_start = max(0, x - padding)
            x_end = min(img.shape[1], x + w + padding)
            
            digit_crop = inverted_img[y_start:y_end, x_start:x_end]
            
            digit_crops.append({'crop': digit_crop, 'x': x, 'y': y})

    digit_crops.sort(key=lambda d: (d['y'], d['x']))
    
    digit_count = 0
    for i, item in enumerate(digit_crops):
        crop = item['crop']
        final_digit = cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        filename = os.path.join(OUTPUT_ROOT_DIR, f'digit_{digit_count:03d}.png')
        cv2.imwrite(filename, final_digit)
        digit_count += 1
        
    print(f"\nSegmentation complete! Saved {digit_count} potential digits using contour detection.")


if __name__ == '__main__':
    segment_and_save_digits()