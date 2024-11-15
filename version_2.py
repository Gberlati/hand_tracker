import cv2
import torch
from version_3 import HandTracker, HandDataset
import os

def record_training_data():
    """Record video for training data"""
    print("Recording training data...")
    print("Press 'r' to start/stop recording, 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    out = None
    recording = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Show recording status
        status = "RECORDING" if recording else "NOT RECORDING"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                   (0, 0, 255) if recording else (255, 255, 255), 2)
        
        cv2.imshow('Recording Training Data', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if not recording:
                # Start recording
                out = cv2.VideoWriter('training_data.mp4', 
                                    cv2.VideoWriter_fourcc(*'mp4v'), 
                                    30, 
                                    (frame.shape[1], frame.shape[0]))
                recording = True
                print("Started recording...")
            else:
                # Stop recording
                out.release()
                recording = False
                print("Stopped recording...")
        
        elif key == ord('q'):
            break
            
        if recording and out is not None:
            out.write(frame)
    
    if out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Recording completed!")

def train_model():
    """Train the hand tracker model"""
    print("Training model...")
    tracker = HandTracker()
    tracker.train("training_data.mp4", epochs=10)
    print("Training completed!")

def run_tracker():
    """Run the trained hand tracker"""
    print("Running hand tracker...")
    print("Press 'q' to quit")
    
    tracker = HandTracker(model_path='hand_model.pth')
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, distance = tracker.process_frame(frame)
        cv2.imshow('Hand Distance Tracker', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    while True:
        print("\nHand Tracker Menu:")
        print("1. Record training data")
        print("2. Train model")
        print("3. Run tracker")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            record_training_data()
        elif choice == '2':
            train_model()
        elif choice == '3':
            run_tracker()
        elif choice == '4':
            break
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()

