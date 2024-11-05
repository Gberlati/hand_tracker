# hand_tracker.py
import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple

class HandTracker:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Tracking state
        self.distance_history = []
        self.smoothing_window = 5
        self.distance_threshold = 0.3  # Normalized distance for warnings
    
    def compute_hand_distance(self, hand_landmarks) -> float:
        """Compute distance between hand centroids."""
        centroids = []
        for hand in hand_landmarks:
            points = np.array([[lm.x, lm.y] for lm in hand.landmark])
            centroids.append(np.mean(points, axis=0))
        return np.linalg.norm(centroids[0] - centroids[1])
    
    def process_frame(self, frame) -> Tuple[float, bool]:
        """Process frame and return distance and warning flag."""
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        # Draw hands and compute distance
        if results.multi_hand_landmarks:
            # Draw landmarks for all detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 255, 0))
                )
            
            # Compute distance if two hands are detected
            if len(results.multi_hand_landmarks) == 2:
                distance = self.compute_hand_distance(results.multi_hand_landmarks)
                
                # Smooth distance
                self.distance_history.append(distance)
                if len(self.distance_history) > self.smoothing_window:
                    self.distance_history.pop(0)
                smoothed_distance = np.mean(self.distance_history)
                
                # Check if hands are too close
                warning = smoothed_distance < self.distance_threshold
                
                # Draw distance and warning
                cv2.putText(
                    frame,
                    f"Distance: {smoothed_distance:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if not warning else (0, 0, 255),
                    2
                )
                
                if warning:
                    cv2.putText(
                        frame,
                        "Warning: Hands too close!",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                
                return smoothed_distance, warning
        
        return None, False

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        distance, warning = tracker.process_frame(frame)
        
        # Show frame
        cv2.imshow('Hand Distance Tracker', frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
