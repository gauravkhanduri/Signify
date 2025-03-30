import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def load_sign_model():
    try:
        X = np.load('X_keypoints.npy')
        y = np.load('y_labels.npy')
        labels = np.load('label_map.npy')
        
        # Print available labels for debugging
        print("Available signs:", labels)
        
        # Determine optimal k based on sample size
        n_samples = len(X)
        if n_samples < 2:
            print("Error: Not enough training samples. Please add more training images.")
            return None, None
            
        k = min(3, n_samples - 1)  # Use k=1 for very small datasets
        
        # Train classifier with adjusted parameters
        model = KNeighborsClassifier(n_neighbors=k, weights='distance')
        model.fit(X, y)
        
        print(f"Model trained with {n_samples} samples using k={k}")
        return model, labels
    except FileNotFoundError:
        print("Please train the model with weather signs first!")
        return None, None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def process_hand_landmarks(hand_landmarks):
    keypoints = []
    for lm in hand_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z])
    return keypoints

def get_weather_description(label):
    # Map signs to weather descriptions (make case-insensitive)
    weather_descriptions = {
        'SUNNY': 'It will be a sunny day!',
        'sunny': 'It will be a sunny day!',
        'Sunny': 'It will be a sunny day!',
        'RAINY': 'Expect rain today',
        'CLOUDY': 'It will be cloudy',
        'STORMY': 'Storm is coming',
        'HOT': 'It will be hot today',
        'COLD': 'It will be cold today',
        'WINDY': 'Expect wind today'
    }
    desc = weather_descriptions.get(label, f"Sign detected: {label}")
    print(f"Label: {label}, Description: {desc}")  # Debug output
    return desc

def detect_weather_signs():
    model, labels = load_sign_model()
    if model is None or labels is None:
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # Changed to detect up to 2 hands
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Convert to RGB and process
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Draw hand landmarks and detect signs
            if results.multi_hand_landmarks:
                # Get hand labels (Left/Right) if available
                hand_labels = []
                if results.multi_handedness:
                    hand_labels = [hand.classification[0].label for hand in results.multi_handedness]

                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS)
                    
                    # Get keypoints and predict sign
                    keypoints = process_hand_landmarks(hand_landmarks)
                    keypoints = np.array(keypoints).reshape(1, -1)
                    
                    try:
                        # Predict the sign using the model
                        prediction = model.predict(keypoints)[0]
                        detected_label = str(labels[prediction]).strip()
                        confidence = np.max(model.predict_proba(keypoints)[0])
                        
                        # Get hand side (Left/Right)
                        hand_side = hand_labels[idx] if idx < len(hand_labels) else ""
                        
                        # Add weather description with confidence
                        weather_desc = get_weather_description(detected_label)
                        weather_desc = f"{hand_side}: {weather_desc}" if hand_side else weather_desc
                        weather_desc += f" ({confidence:.2%})"
                        
                        # Position text based on which hand (left/right)
                        y_pos = 30 + (idx * 40)  # Stack text vertically for multiple hands
                        cv2.putText(frame, weather_desc, (10, y_pos),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Prediction error: {str(e)}")
                        continue

            # Show live feed
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Weather Sign Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_weather_signs()
