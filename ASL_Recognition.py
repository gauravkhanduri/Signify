import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import json
import time
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import base64

# Initialize Flask app
app = Flask(__name__, 
    static_url_path='',
    static_folder='static',
    template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Path to the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'asl_model.h5')

# Load the ASL recognition model
class ASLRecognizer:
    def __init__(self, model_path=None):
        self.model = None
        self.labels = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'SPACE', 'DELETE', 'NOTHING'
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"Model not found at {model_path}, using placeholder predictions")
    
    def load_model(self, model_path):
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_landmarks(self, landmarks):
        """Convert landmarks to a flat array suitable for model input"""
        data = []
        
        for landmark in landmarks:
            # Extract x, y, z coordinates
            data.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array([data])
    
    def predict(self, landmarks):
        """Predict ASL sign from hand landmarks"""
        # If model isn't loaded, use a placeholder prediction for development
        if self.model is None:
            # Simple development placeholder - return random letters occasionally
            # In production, you would use your trained model here
            if np.random.random() < 0.1:  # Only return predictions occasionally to simulate real usage
                pred_idx = np.random.randint(0, len(self.labels) - 3)  # Exclude space/delete/nothing
                return {
                    'label': self.labels[pred_idx],
                    'confidence': np.random.uniform(0.7, 0.95)
                }
            return {'label': 'NOTHING', 'confidence': 0.99}
        
        # Process the landmarks for the model
        processed_data = self.preprocess_landmarks(landmarks)
        
        # Make prediction
        predictions = self.model.predict(processed_data)[0]
        
        # Get the highest confidence prediction
        pred_idx = np.argmax(predictions)
        confidence = float(predictions[pred_idx])
        
        return {
            'label': self.labels[pred_idx],
            'confidence': confidence
        }

# Initialize the recognizer
recognizer = ASLRecognizer(MODEL_PATH if os.path.exists(MODEL_PATH) else None)

# Global variables for gesture tracking
last_gesture = None
gesture_start_time = None
recognized_text = ""
confidence_threshold = 0.7
gesture_hold_time = 1.0  # seconds

def process_frame(frame):
    global last_gesture, gesture_start_time, recognized_text
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and find hand landmarks
    results = hands.process(rgb_frame)
    
    # Draw landmarks and get prediction
    annotated_frame = frame.copy()
    prediction = {'label': 'NOTHING', 'confidence': 0.0}
    
    if results.multi_hand_landmarks:
        # Get the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(
            annotated_frame, 
            hand_landmarks, 
            mp_hands.HAND_CONNECTIONS
        )
        
        # Get prediction from the model
        prediction = recognizer.predict(hand_landmarks.landmark)
        
        # Process the prediction
        if prediction['confidence'] >= confidence_threshold:
            if prediction['label'] != 'NOTHING':
                # If this is a new gesture
                if last_gesture != prediction['label']:
                    last_gesture = prediction['label']
                    gesture_start_time = time.time()
                # If the same gesture is being held
                elif (time.time() - gesture_start_time) >= gesture_hold_time:
                    # Add the recognized character to our text
                    if prediction['label'] == 'SPACE':
                        recognized_text += ' '
                    elif prediction['label'] == 'DELETE':
                        recognized_text = recognized_text[:-1] if recognized_text else ""
                    else:
                        recognized_text += prediction['label']
                    
                    # Reset the gesture timing
                    gesture_start_time = time.time()
                    
                # Display the current gesture
                cv2.putText(
                    annotated_frame,
                    f"Sign: {prediction['label']} ({prediction['confidence']:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            else:
                last_gesture = None
        else:
            last_gesture = None
    else:
        last_gesture = None
    
    # Display the recognized text
    cv2.putText(
        annotated_frame,
        f"Text: {recognized_text}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )
    
    return annotated_frame, {
        'prediction': prediction,
        'text': recognized_text
    }

# API Routes
@app.route('/process_image', methods=['POST'])
def process_image_route():
    if 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400
    
    # Decode the base64 image
    try:
        base64_image = request.json['image'].split(',')[1] if ',' in request.json['image'] else request.json['image']
        image_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
    except Exception as e:
        return jsonify({'error': f'Error decoding image: {str(e)}'}), 400
    
    # Process the frame
    try:
        annotated_frame, results = process_frame(frame)
        
        # Encode the annotated frame back to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        annotated_image = base64.b64encode(buffer).decode('utf-8')
        
        # Return results and the annotated image
        return jsonify({
            'annotated_image': f'data:image/jpeg;base64,{annotated_image}',
            'prediction': results['prediction'],
            'text': results['text']
        })
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/save_frame', methods=['POST'])
def save_frame():
    try:
        data = request.json
        base64_image = data['image'].split(',')[1]
        filename = data['filename']
        
        # Ensure directory exists
        save_dir = os.path.join('data', 'today')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save image
        image_data = base64.b64decode(base64_image)
        file_path = os.path.join(save_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(image_data)
            
        return jsonify({'success': True, 'path': file_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset_text', methods=['POST'])
def reset_text():
    global recognized_text
    recognized_text = ""
    return jsonify({'success': True, 'text': recognized_text})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

# Add root route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.static_folder, 'favicon.ico')

# For local testing
def test_camera():
    cap = cv2.VideoCapture(0)
    global recognized_text
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        annotated_frame, _ = process_frame(frame)
        
        cv2.imshow('ASL Recognition', annotated_frame)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ASL Recognition Backend')
    parser.add_argument('--test', action='store_true', help='Run camera test')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()
    
    # Create required directories if they don't exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    if args.test:
        print("Running camera test mode...")
        test_camera()
    else:
        print(f"Starting server on {args.host}:{args.port}...")
        app.run(host=args.host, port=args.port, debug=True)