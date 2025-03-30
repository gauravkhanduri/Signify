import asyncio
import websockets
import cv2
import numpy as np
import base64
import mediapipe as mp
import json

class SignDetector:
    def __init__(self):
        # Load the trained model and MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7)
        
        # Frame tracking
        self.frame_count = 0
        self.total_frames = 24
        self.detections = []
        self.unique_signs = []  # Track unique signs in order of appearance
        
        # Simple sign detection based on hand position
        self.sign_positions = {
            'how': {'y_threshold': 0.3},    # Hand raised high
            'weather': {'y_threshold': 0.5}, # Hand in middle
            'today': {'y_threshold': 0.7}    # Hand lower
        }
        print("\n[SYSTEM] Sign detector initialized")
        print("[SYSTEM] Using thresholds:")
        print(f"  HOW: y < {self.sign_positions['how']['y_threshold']}")
        print(f"  WEATHER: y < {self.sign_positions['weather']['y_threshold']}")
        print(f"  TODAY: y >= {self.sign_positions['weather']['y_threshold']}")

    async def process_frame(self, frame_data):
        try:
            self.frame_count += 1
            print(f"\n[FRAME {self.frame_count}/{self.total_frames}]")

            # Decode base64 image
            try:
                img_data = base64.b64decode(frame_data.split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                print("[DEBUG] Frame decoded successfully")
            except Exception as e:
                print(f"[ERROR] Frame decode error: {e}")
                return {'status': 'error', 'message': 'Frame decode error'}

            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                hand_y = sum(lm.y for lm in results.multi_hand_landmarks[0].landmark) / 21
                print(f"[DEBUG] Hand Y position: {hand_y:.3f}")
                
                detected_sign = None
                if hand_y < self.sign_positions['how']['y_threshold']:
                    detected_sign = 'how'
                elif hand_y < self.sign_positions['weather']['y_threshold']:
                    detected_sign = 'weather'
                else:
                    detected_sign = 'today'
                
                # Track unique signs in order
                if detected_sign not in self.unique_signs:
                    self.unique_signs.append(detected_sign)
                
                self.detections.append(detected_sign)
                print(f"[DETECTED] Frame {self.frame_count}: {detected_sign}")
                
                response = {
                    'status': 'success',
                    'frame': self.frame_count,
                    'frame_total': self.total_frames,
                    'detected_sign': detected_sign,
                    'unique_signs': self.unique_signs
                }
                
                # If all frames processed, include final sequence
                if self.frame_count == self.total_frames:
                    print("\n=== FINAL SEQUENCE ===")
                    print(" → ".join(self.unique_signs))
                    response['complete'] = True
                    response['final_sequence'] = self.unique_signs
                
                print(f"[DEBUG] Sending response: {response}")
                return response

            print("[NO HAND] No hand detected in frame")
            return {
                'status': 'no_hand',
                'frame': self.frame_count
            }
            
        except Exception as e:
            print(f"[ERROR] Frame {self.frame_count}: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def reset(self):
        print("\n[SYSTEM] Resetting frame counter")
        self.frame_count = 0
        self.detections = []
        self.unique_signs = []
        print("\n[RESET] Ready for next sequence")

async def websocket_handler(websocket):
    detector = SignDetector()
    print("\n[INFO] New client connected")
    print("[INFO] Ready to process frames...")
    
    try:
        async for message in websocket:
            result = await detector.process_frame(message)
            await websocket.send(json.dumps(result))
            
            if detector.frame_count >= detector.total_frames:
                print(f"\n[COMPLETE] Processed all {detector.total_frames} frames")
                detector.reset()
                
    except websockets.exceptions.ConnectionClosed:
        print("[INFO] Client disconnected")

async def main():
    try:
        server = await websockets.serve(
            websocket_handler,
            "localhost",
            8765
        )
        print("Sign detection server running on ws://localhost:8765")
        await server.wait_closed()
    except OSError as e:
        print(f"Error starting server: {e}")
        print("Try using a different port or ensure no other servers are running")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
