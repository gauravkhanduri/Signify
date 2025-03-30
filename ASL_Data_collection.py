import cv2
import numpy as np
import mediapipe as mp
import os
import time
import pandas as pd
import argparse
from tqdm import tqdm

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Constants
DATA_DIR = 'data/asl_dataset'
LABELS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['SPACE', 'DELETE', 'NOTHING']
SAMPLES_PER_LABEL = 100
COUNTDOWN_TIME = 3  # seconds
CAPTURE_TIME = 5    # seconds
LANDMARK_COUNT = 21  # MediaPipe provides 21 hand landmarks

def setup_data_directory():
    """Create the data directory if it doesn't exist"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    print(f"Data will be saved to {DATA_DIR}")

def collect_samples(label, num_samples):
    """Collect hand landmark samples for a specific label"""
    print(f"Collecting samples for: {label}")
    
    cap = cv2.VideoCapture(0)
    
    # Wait for camera to initialize
    time.sleep(1)
    
    # Prepare data structures
    samples = []
    
    # Start capture countdown
    start_time = time.time()
    countdown_end = start_time + COUNTDOWN_TIME
    capture_end = countdown_end + CAPTURE_TIME
    
    while time.time() < capture_end:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera")
            break
        
        # Show instructions
        current_time = time.time()
        if current_time < countdown_end:
            # In countdown phase
            remaining = int(countdown_end - current_time)
            cv2.putText(
                frame,
                f"Get ready to show '{label}' in {remaining}...",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        else:
            # In capture phase
            progress = int(((current_time - countdown_end) / CAPTURE_TIME) * 100)
            cv2.putText(
                frame,
                f"Show '{label}' - Capturing: {progress}%",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Process the frame and extract landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                # Draw landmarks on the frame
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                
                # Extract the first hand's landmarks
                landmarks = results.multi_hand_landmarks[0].landmark
                
                # Create a flat list of the landmark coordinates
                landmark_data = []
                for lm in landmarks:
                    landmark_data.extend([lm.x, lm.y, lm.z])
                
                # Add the sample with the label
                samples.append(landmark_data)
                
                # Show that a sample was captured
                cv2.putText(
                    frame,
                    f"Samples: {len(samples)}/{num_samples}",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )
                
                # If we have enough samples, we can stop
                if len(samples) >= num_samples:
                    break
        
        # Display the frame
        cv2.imshow('ASL Data Collection', frame)
        
        # Break on ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Collected {len(samples)} samples for {label}")
    return samples

def save_data(all_samples, labels):
    """Save the collected data to a CSV file"""
    # Create column names
    columns = []
    for i in range(LANDMARK_COUNT):
        columns.extend([f'x{i}', f'y{i}', f'z{i}'])
    columns.append('label')
    
    # Convert to DataFrame
    data = []
    for label, samples in zip(labels, all_samples):
        for sample in samples:
            row = sample + [label]
            data.append(row)
    
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    csv_path = os.path.join(DATA_DIR, 'asl_landmarks.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Data saved to {csv_path}")
    print(f"Total samples: {len(df)}")

def main():
    parser = argparse.ArgumentParser(description='Collect ASL hand landmark data')
    parser.add_argument('--labels', nargs='+', choices=LABELS, default=LABELS,
                      help='Specific labels to collect data for')
    parser.add_argument('--samples', type=int, default=SAMPLES_PER_LABEL,
                      help='Number of samples to collect per label')
    args = parser.parse_args()
    
    setup_data_directory()
    
    all_samples = []
    collected_labels = []
    
    for label in tqdm(args.labels, desc="Collecting data for labels"):
        samples = collect_samples(label, args.samples)
        if samples:
            all_samples.append(samples)
            collected_labels.append(label)
    
    if all_samples:
        save_data(all_samples, collected_labels)
    else:
        print("No data was collected")

if __name__ == "__main__":
    main()