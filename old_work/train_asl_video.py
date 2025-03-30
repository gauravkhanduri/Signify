import cv2
import mediapipe as mp
import numpy as np
import os

def extract_video_keypoints(video_path):
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(video_path)
    sequence = []
    expected_points = 126  # 21 landmarks × 3 coordinates × 2 hands
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            # Initialize frame keypoints with zeros
            frame_keypoints = np.zeros(expected_points)
            
            if results.multi_hand_landmarks:
                point_idx = 0
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        if point_idx + 2 < expected_points:  # Ensure we don't exceed array bounds
                            frame_keypoints[point_idx] = lm.x
                            frame_keypoints[point_idx + 1] = lm.y
                            frame_keypoints[point_idx + 2] = lm.z
                            point_idx += 3
                
                sequence.append(frame_keypoints)
    
    cap.release()
    
    if len(sequence) == 0:
        return np.array([[]])
        
    return np.array(sequence)

def train_asl_model(video_dir):
    word_sequences = {
        'how': [],
        'weather': [],
        'today': []
    }
    
    # Process each word's video
    for video_file in os.listdir(video_dir):
        if video_file.endswith('.mp4'):
            word = video_file.split('.')[0].lower()
            if word in word_sequences:
                video_path = os.path.join(video_dir, video_file)
                sequence = extract_video_keypoints(video_path)
                
                if sequence.shape[0] > 1:  # Check if we got valid sequences
                    word_sequences[word].append(sequence)
                    print(f"Processed {word} video: {len(sequence)} frames")
                else:
                    print(f"Warning: No valid frames in {video_file}")
    
    # Create training data from word combinations
    X = []
    y = []
    
    # Combine sequences to create full sentences
    for how_seq in word_sequences['how']:
        for weather_seq in word_sequences['weather']:
            for today_seq in word_sequences['today']:
                # Normalize sequence lengths
                target_len = min(len(how_seq), len(weather_seq), len(today_seq))
                if target_len < 5:  # Skip if sequences are too short
                    continue
                    
                # Resize sequences to target length
                how_resized = how_seq[:target_len]
                weather_resized = weather_seq[:target_len]
                today_resized = today_seq[:target_len]
                
                # Create full sentence sequence
                full_sequence = np.concatenate([
                    how_resized,
                    weather_resized,
                    today_resized
                ], axis=1)
                
                X.append(full_sequence)
                y.append(1)  # 1 for valid sentence
    
    if len(X) > 0:
        X = np.array(X)
        y = np.array(y)
        
        # Save the processed data
        np.save('X_video_keypoints.npy', X)
        np.save('y_video_labels.npy', y)
        
        print(f"Created {len(X)} training sequences")
        print(f"Sequence shape: {X[0].shape}")
    else:
        print("No valid sequences created. Check video files.")

if __name__ == "__main__":
    VIDEO_DIR = "asl_videos"
    required_files = ['how.mp4', 'weather.mp4', 'today.mp4']
    
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)
        print(f"Created directory: {VIDEO_DIR}")
        print(f"Please add these videos: {', '.join(required_files)}")
    else:
        # Check for required files
        missing = [f for f in required_files 
                  if not os.path.exists(os.path.join(VIDEO_DIR, f))]
        if missing:
            print(f"Missing required videos: {', '.join(missing)}")
        else:
            train_asl_model(VIDEO_DIR)
