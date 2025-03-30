import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from sklearn.neighbors import KNeighborsClassifier
import time
import os

class ASLRecognizer:
    def __init__(self):
        try:
            # Check if model files exist
            required_files = ['X_video_keypoints.npy', 'y_video_labels.npy']
            for file in required_files:
                if not os.path.exists(file):
                    raise FileNotFoundError(f"Missing {file}. Run train_asl_video.py first.")

            # Load video sequence model
            self.X_video = np.load('X_video_keypoints.npy')
            self.y_video = np.load('y_video_labels.npy')
            
            print(f"Loaded {len(self.X_video)} sequences for training")
            print(f"Sequence shape: {self.X_video.shape}")
            
            # Initialize buffer with correct size based on training data
            self.sequence_length = self.X_video.shape[1]  # Get expected sequence length
            self.num_features = self.X_video.shape[2]     # Get number of features per frame
            self.sequence_buffer = deque(maxlen=self.sequence_length)
            
            # Setup MediaPipe
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7)
            
            # Train sequence classifier
            print("Training sequence classifier...")
            self.sequence_model = KNeighborsClassifier(n_neighbors=1)
            X_flat = self.X_video.reshape(len(self.X_video), -1)
            self.sequence_model.fit(X_flat, self.y_video)
            print("Classifier trained successfully")
            
            # Add gesture sequence tracking
            self.expected_sequence = ['how', 'weather', 'today']
            self.current_sequence = []
            self.last_gesture = None
            self.gesture_cooldown = 30  # Frames to wait between gestures
            self.cooldown_counter = 0
            self.confidence_threshold = 0.7
            
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def predict_gesture(self, frame_keypoints):
        try:
            sequence = np.array([frame_keypoints])
            sequence_flat = sequence.flatten().reshape(1, -1)
            prediction = self.sequence_model.predict(sequence_flat)[0]
            confidence = np.max(self.sequence_model.predict_proba(sequence_flat)[0])
            
            if confidence > self.confidence_threshold:
                return prediction, confidence
            return None, 0
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None, 0

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        # Draw current sequence progress
        y_base = 30
        cv2.putText(frame, f"Expected: {' -> '.join(self.expected_sequence)}", 
                  (10, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.current_sequence:
            cv2.putText(frame, f"Current: {' -> '.join(self.current_sequence)}", 
                      (10, y_base + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                if self.cooldown_counter <= 0:
                    # Extract and predict gesture
                    keypoints = []
                    for lm in hand_landmarks.landmark:
                        keypoints.extend([lm.x, lm.y, lm.z])
                    
                    # Pad for consistent feature count
                    while len(keypoints) < self.num_features:
                        keypoints.extend([0.0])
                    
                    prediction, confidence = self.predict_gesture(keypoints)
                    
                    if prediction is not None:
                        gesture = self.expected_sequence[prediction]
                        expected_idx = len(self.current_sequence)
                        
                        # Only accept gesture if it's the next expected one
                        if expected_idx < len(self.expected_sequence) and gesture == self.expected_sequence[expected_idx]:
                            self.current_sequence.append(gesture)
                            self.cooldown_counter = self.gesture_cooldown
                            
                            # Display the detected gesture
                            cv2.putText(frame, f"Detected: {gesture} ({confidence:.2f})", 
                                      (10, y_base + 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                      1, (0, 255, 0), 2)
        
        # Update cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        
        # Check if sequence is complete
        if len(self.current_sequence) == len(self.expected_sequence):
            cv2.putText(frame, "Complete: How's the weather today?", 
                      (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (0, 255, 255), 2)
            # Reset after displaying for a moment
            if self.cooldown_counter <= 0:
                self.current_sequence = []
        
        return frame

    def predict_sequence(self, sequence):
        sequence_flat = sequence.flatten().reshape(1, -1)
        return self.sequence_model.predict(sequence_flat)[0]

    def __del__(self):
        self.hands.close()

def main():
    recognizer = ASLRecognizer()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame for ASL recognition
        processed_frame = recognizer.process_frame(frame)
        
        cv2.putText(processed_frame, "Press 'q' to quit", 
                    (10, processed_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('ASL Recognition', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
