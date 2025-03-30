import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

def create_sign_folders():
    signs = ['how', 'weather', 'today']
    data_dir = 'data'
    
    for sign in signs:
        path = os.path.join(data_dir, sign)
        os.makedirs(path, exist_ok=True)
    return signs

def collect_sign_data():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
    
    cap = cv2.VideoCapture(0)
    signs = create_sign_folders()
    samples_per_sign = 20  # Number of samples to collect per sign
    
    for sign in signs:
        collected = 0
        print(f"\nCollecting samples for '{sign.upper()}' sign...")
        print("Press 'c' to capture, 'n' for next sign, 'q' to quit")
        
        while collected < samples_per_sign:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Display instructions and progress
            cv2.putText(frame, f"Sign: {sign.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {collected}/{samples_per_sign}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Collect Signs', frame)
            key = cv2.waitKey(1)
            
            if key == ord('c') and results.multi_hand_landmarks:
                # Save frame
                filename = f"{sign}_{collected}.jpg"
                filepath = os.path.join('data', sign, filename)
                cv2.imwrite(filepath, frame)
                collected += 1
                print(f"Saved {filename}")
            
            elif key == ord('n'):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return False
    
    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    print("Starting ASL Weather Signs Data Collection...")
    print("Will collect samples for: HOW, WEATHER, TODAY")
    if collect_sign_data():
        print("\nData collection completed successfully!")
        print("You can now run extract_keypoints.py to process the collected data.")
