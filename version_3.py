import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
import os

class HandDetectionNet(nn.Module):
    def __init__(self):
        super(HandDetectionNet, self).__init__()
        # CNN for processing image
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
        )
        
        # Fully connected layers for hand keypoints
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 84)  # 42 keypoints (x,y) = 84 values
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        keypoints = self.fc_layers(x)
        return keypoints  # Shape: (batch_size, 84)

class HandDataset(Dataset):
    def __init__(self, video_path=None, transform=None):
        self.transform = transform
        self.data = []
        
        if video_path:
            self.generate_data(video_path)
    
    def generate_data(self, video_path):
        # Initialize MediaPipe for data generation
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame and get landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                # Extract landmarks
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_points = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
                    landmarks.extend(hand_points)
                
                # Flatten landmarks into a single array
                landmarks_flat = np.array(landmarks).flatten()
                
                # Store frame and landmarks
                self.data.append({
                    'image': cv2.resize(frame, (224, 224)),
                    'landmarks': landmarks_flat.astype(np.float32)
                })
        
        cap.release()
        print(f"Generated {len(self.data)} training samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample['image']
        landmarks = sample['landmarks']
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': torch.FloatTensor(image).permute(2, 0, 1) / 255.0,
            'landmarks': torch.FloatTensor(landmarks)  # Shape: (84,)
        }

class HandTracker:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HandDetectionNet().to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.distance_history = []
        self.smoothing_window = 5
    
    def train(self, video_path, epochs=10):
        # Create dataset
        dataset = HandDataset(video_path)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                images = batch['image'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)  # Shape: (batch_size, 84)
                
                # Forward pass
                outputs = self.model(images)  # Shape: (batch_size, 84)
                loss = criterion(outputs, landmarks)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        torch.save(self.model.state_dict(), 'hand_model.pth')
    
    def process_frame(self, frame):
        # Preprocess frame
        frame = cv2.resize(frame, (224, 224))
        input_tensor = torch.FloatTensor(frame).permute(2, 0, 1).unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            landmarks = self.model(input_tensor)
            landmarks = landmarks.cpu().numpy()[0].reshape(-1, 2)  # Shape: (42, 2)
        
        # Split landmarks into two hands
        hand1_landmarks = landmarks[:21]  # First 21 landmarks
        hand2_landmarks = landmarks[21:]  # Last 21 landmarks
        
        # Compute hand centers
        hand1_center = hand1_landmarks.mean(axis=0)
        hand2_center = hand2_landmarks.mean(axis=0)
        
        # Calculate distance
        distance = np.linalg.norm(hand1_center - hand2_center)
        
        # Smooth distance
        self.distance_history.append(distance)
        if len(self.distance_history) > self.smoothing_window:
            self.distance_history.pop(0)
        smoothed_distance = np.mean(self.distance_history)
        
        # Draw predictions
        frame = self.draw_predictions(frame, hand1_landmarks, hand2_landmarks, smoothed_distance)
        return frame, smoothed_distance
    
    def draw_predictions(self, frame, hand1_landmarks, hand2_landmarks, distance):
        # Draw landmarks for both hands
        for points in [hand1_landmarks, hand2_landmarks]:
            for point in points:
                x, y = int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        # Draw distance
        cv2.putText(
            frame,
            f"Distance: {distance:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        return frame
