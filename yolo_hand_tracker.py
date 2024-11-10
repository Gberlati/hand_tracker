import cv2
import mediapipe as mp
import torch
import numpy as np
import math
import asyncio
import websockets
import json
import threading
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
current_distance = 0
websocket_connections = set()

# WebSocket handler with proper signature
async def websocket_handler(websocket):
    try:
        websocket_connections.add(websocket)
        logger.info("New WebSocket connection established")
        while True:
            try:
                await websocket.send(json.dumps({'distance': current_distance}))
                await asyncio.sleep(0.016)
            except websockets.exceptions.ConnectionClosed:
                break
    except Exception as e:
        logger.error(f"Error in websocket handler: {e}")
    finally:
        websocket_connections.remove(websocket)
        logger.info("WebSocket connection closed")

# WebSocket server
async def run_server():
    try:
        async with websockets.serve(websocket_handler, 'localhost', 8765, ping_interval=None):
            logger.info("WebSocket server started on ws://localhost:8765")
            await asyncio.Future()  # run forever
    except Exception as e:
        logger.error(f"Error starting WebSocket server: {e}")

# Start WebSocket server in background
def start_websocket_server():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_server())
    except Exception as e:
        logger.error(f"Error in websocket server thread: {e}")

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_hand_center(landmarks, frame_shape):
    h, w = frame_shape[:2]
    x_coords = [lm.x * w for lm in landmarks.landmark]
    y_coords = [lm.y * h for lm in landmarks.landmark]
    return (int(np.mean(x_coords)), int(np.mean(y_coords)))

def main():
    global current_distance
    
    try:
        # Start WebSocket server in a separate thread
        server_thread = threading.Thread(target=start_websocket_server, daemon=True)
        server_thread.start()
        logger.info("WebSocket server thread started")

        # Initialize MediaPipe
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            logger.error("Could not open webcam")
            return

        # Check CUDA availability
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA not available. Using CPU.")
            device = torch.device("cpu")

        AVERAGE_HAND_WIDTH_CM = 10.0

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            hand_centers = []
            
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(0,0,255), thickness=2)
                    )
                    
                    h, w, _ = frame.shape
                    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                    
                    x1, y1 = int(min(x_coords)), int(min(y_coords))
                    x2, y2 = int(max(x_coords)), int(max(y_coords))
                    hand_width_pixels = x2 - x1
                    
                    padding = 20
                    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                    x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    center = get_hand_center(hand_landmarks, frame.shape)
                    hand_centers.append((center, hand_width_pixels))
                    cv2.circle(frame, center, 5, (255, 0, 0), -1)
                    
                    cv2.putText(
                        frame,
                        f"Hand {idx + 1}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                if len(hand_centers) == 2:
                    cv2.line(frame, hand_centers[0][0], hand_centers[1][0], (255, 0, 0), 2)
                    pixel_distance = calculate_distance(hand_centers[0][0], hand_centers[1][0])
                    avg_hand_width_pixels = (hand_centers[0][1] + hand_centers[1][1]) / 2
                    pixels_per_cm = avg_hand_width_pixels / AVERAGE_HAND_WIDTH_CM
                    real_distance_cm = pixel_distance / pixels_per_cm
                    
                    current_distance = real_distance_cm
                    
                    mid_point = (
                        (hand_centers[0][0][0] + hand_centers[1][0][0]) // 2,
                        (hand_centers[0][0][1] + hand_centers[1][0][1]) // 2 - 30
                    )
                    cv2.putText(
                        frame,
                        f"Distance: {real_distance_cm:.1f} cm",
                        mid_point,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2
                    )

            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow('Hand Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()