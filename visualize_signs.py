import numpy as np
import cv2
import mediapipe as mp
import os

def load_and_visualize():
    try:
        # Load saved data
        X = np.load('X_keypoints.npy')
        y = np.load('y_labels.npy')
        labels = np.load('label_map.npy')
        
        print(f"Loaded {len(X)} samples with labels: {labels}")
    except FileNotFoundError as e:
        print("Error: Could not find saved keypoint files.")
        print("Please run extract_keypoints.py first to generate the data files.")
        return
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # Create a blank image
    img_size = 400
    
    for i in range(len(X)):
        image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Convert normalized coordinates to image coordinates
        landmarks = []
        for j in range(0, len(X[i]), 3):
            x, y_coord = int(X[i][j] * img_size), int(X[i][j+1] * img_size)
            landmarks.append((x, y_coord))
            cv2.circle(image, (x, y_coord), 3, (0, 255, 0), -1)
        
        # Draw connections
        connections = mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            cv2.line(image, landmarks[start_idx], landmarks[end_idx], (255, 255, 255), 1)
        
        # Get label directly from y array and use it to index labels array
        try:
            label_idx = y[i]  # y[i] is already an integer
            label = str(labels[label_idx]) if 0 <= label_idx < len(labels) else f"Unknown ({label_idx})"
        except Exception as e:
            print(f"Error getting label for sample {i}: {e}")
            label = f"Error ({i})"
        
        cv2.putText(image, f"Sign: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add instructions
        cv2.putText(image, "Press ESC to exit, any key for next", 
                    (10, img_size - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)
        
        cv2.imshow('Saved Hand Sign', image)
        key = cv2.waitKey(0)
        if key == 27:  # ESC key
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    load_and_visualize()
