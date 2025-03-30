import cv2
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm
from collections import deque, OrderedDict
import asyncio
import json

try:
    import websockets
except ImportError:
    print("WebSocket support requires additional packages.")
    print("Please install required packages using:")
    print("pip install websockets")
    print("\nAfter installation, run this script again.")
    exit(1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

DATA_DIR = 'data'
# Check if the path exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Directory '{DATA_DIR}' not found")

X = []
y = []

# Get all files in data directory
all_files = os.listdir(DATA_DIR)

# Add expected signs
EXPECTED_SIGNS = ['how', 'weather', 'today']
print(f"Processing signs: {EXPECTED_SIGNS}")

sign_sequence = []  # Store detected signs in order
SEQUENCE_WINDOW = 3  # How many recent signs to track
detected_sequences = OrderedDict()  # Use OrderedDict to maintain order but prevent duplicates

seen_signs = set()  # Track unique signs
current_sequence = []  # Track sequence in order of first appearance

ws_server = None
connected_clients = set()

async def send_to_clients(message):
    if connected_clients:
        await asyncio.gather(
            *[client.send(json.dumps(message)) for client in connected_clients]
        )

async def websocket_handler(websocket):  # Remove path parameter
    connected_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.remove(websocket)

def visualize_hand(image_path, landmarks=None, label=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if landmarks is None:
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
    
    # Create a copy for visualization
    display_image = image.copy()
    
    if landmarks:
        # Draw the landmarks
        mp_drawing.draw_landmarks(
            display_image,
            landmarks,
            mp_hands.HAND_CONNECTIONS)
    
    # Add label text if provided
    if label:
        cv2.putText(display_image, f"Sign: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Only add to sequence if it's a new sign
        if label not in seen_signs:
            seen_signs.add(label)
            current_sequence.append(label)
            print(f"\nDetected new sign: {label}")
            print(f"Current sequence: {' → '.join(current_sequence)}")
            
            # Save current sequence to file
            with open('unique_sequence.txt', 'w') as f:
                f.write(' → '.join(current_sequence))
            
            # Send to WebSocket clients
            asyncio.get_event_loop().run_until_complete(
                send_to_clients({
                    'detected_sign': label,
                    'sequence': current_sequence
                })
            )
    
    # Display the image
    cv2.imshow('Hand Sign', display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Process files based on structure
for item in all_files:
    item_path = os.path.join(DATA_DIR, item)
    
    # Only process directories for our expected signs
    if os.path.isdir(item_path) and item.lower() in EXPECTED_SIGNS:
        label = item.lower()  # Convert to lowercase for consistency
        image_files = [f for f in os.listdir(item_path) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        image_paths = [os.path.join(item_path, f) for f in image_files]
        
        print(f"\nProcessing {label} sign: {len(image_paths)} images")
        
        # Process each image with progress tracking
        for img_path in tqdm(image_paths, desc=f"Processing {label}"):
            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                    
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Visualize the detection
                    visualize_hand(img_path, hand_landmarks, label)
                    
                    keypoints = []
                    for lm in hand_landmarks.landmark:
                        keypoints.append(lm.x)
                        keypoints.append(lm.y)
                        keypoints.append(lm.z)
                    
                    X.append(keypoints)
                    y.append(label)
                else:
                    print(f"Warning: No hand detected in {img_path}")
                    
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue

hands.close()

if len(X) == 0:
    print("No valid samples were processed!")
    exit(1)

# Convert labels to numerical indices
unique_labels = list(set(y))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
y = [label_to_idx[label] for label in y]

X = np.array(X)
y = np.array(y)
print(f"Extracted {X.shape[0]} samples, each with {X.shape[1]} features.")
print(f"Labels: {unique_labels}")

# At the end, print sign-specific statistics
sign_counts = {}
for label, idx in label_to_idx.items():
    count = np.sum(y == idx)
    sign_counts[label] = count
    print(f"Processed {count} samples for {label}")

print("\nSummary:")
print(f"Total samples: {len(X)}")
print("Samples per sign:")
for sign, count in sign_counts.items():
    print(f"  {sign}: {count}")

# Save to .npy for use in training
np.save('X_keypoints.npy', X)
np.save('y_labels.npy', y)
np.save('label_map.npy', np.array(unique_labels))

print("\nFull sign sequence detected:")
for i in range(0, len(sign_sequence), SEQUENCE_WINDOW):
    sequence_chunk = sign_sequence[i:i+SEQUENCE_WINDOW]
    print(' → '.join(sequence_chunk))

# Save full sequence to file
with open('full_sequence.txt', 'w') as f:
    f.write('\n'.join([' → '.join(sign_sequence[i:i+SEQUENCE_WINDOW]) 
                      for i in range(0, len(sign_sequence), SEQUENCE_WINDOW)]))

print("\nFinal unique sequence in order of appearance:")
print(' → '.join(current_sequence))

# Save final sequence
with open('final_sequence.txt', 'w') as f:
    f.write(' → '.join(current_sequence))

print("\nPress any key to close the visualization windows...")

async def start_server(port=8765):
    global ws_server
    try:
        ws_server = await websockets.serve(
            websocket_handler,
            "localhost", 
            port
        )
        print(f"WebSocket server running on ws://localhost:{port}")
        return True
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"Port {port} is busy, trying alternate port...")
            return await start_server(port + 1)
        else:
            print(f"Error starting server: {e}")
            return False

if __name__ == "__main__":
    # Start WebSocket server
    loop = asyncio.get_event_loop()
    if not loop.run_until_complete(start_server()):
        print("Failed to start WebSocket server")
        exit(1)
    
    loop.run_in_executor(None, loop.run_forever)
    print("\nProcessing complete.")
