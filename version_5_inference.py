import cv2
import torch
import numpy as np
from torchvision import transforms
import time

class PalmDetectionModel(torch.nn.Module):
    def __init__(self, max_boxes=4):
        super(PalmDetectionModel, self).__init__()
        self.max_boxes = max_boxes
        
        # Feature extraction backbone (using ResNet-like architecture)
        self.features = torch.nn.Sequential(
            # Initial conv block
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Residual blocks
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
            self._make_layer(256, 512, 2),
            
            # Final layers
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Regression head for bounding boxes
        self.bbox_head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, max_boxes * 5)  # 5 values per box (4 coords + 1 conf)
        )
    
    def _make_layer(self, in_channels, out_channels, stride):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.bbox_head(features)
        
        batch_size = output.shape[0]
        output = output.view(batch_size, self.max_boxes, 5)
        
        confidence = torch.sigmoid(output[..., 0])
        bbox = output[..., 1:]
        
        return confidence, bbox

class HandDetector:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 confidence_threshold=0.7):  # Increased confidence threshold
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Initialize model
        self.model = PalmDetectionModel(max_boxes=4)
        self.model.to(device)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize transform - exactly as in training
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def apply_nms(self, detections, iou_threshold=0.5):
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if not detections:
            return []

        # Convert detections to format suitable for NMS
        boxes = np.array([[d['bbox'][0], d['bbox'][1], 
                          d['bbox'][0] + d['bbox'][2], 
                          d['bbox'][1] + d['bbox'][3]] for d in detections])
        scores = np.array([d['confidence'] for d in detections])

        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by confidence
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
                
            # Calculate IoU with rest of boxes
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]

    def is_valid_hand(self, bbox, frame_height, frame_width):
        """Check if detection is likely to be a hand based on size and aspect ratio"""
        x, y, w, h = bbox
        
        # Size constraints
        min_size = min(frame_height, frame_width) * 0.05  # Min 5% of frame
        max_size = max(frame_height, frame_width) * 0.4   # Max 40% of frame
        
        if w < min_size or h < min_size:
            return False
        if w > max_size or h > max_size:
            return False
        
        # Aspect ratio constraints for hands (width/height)
        aspect_ratio = w / h
        min_aspect_ratio = 0.3  # Allows for vertical hand orientations
        max_aspect_ratio = 2.0  # Allows for horizontal hand orientations
        
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            return False
        
        return True

    def preprocess_frame(self, frame):
        # [Previous preprocess_frame code remains the same]
        target_size = 224
        h, w = frame.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        frame_resized = cv2.resize(frame, (new_w, new_h))
        square_frame = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        
        square_frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = frame_resized
        frame_rgb = cv2.cvtColor(square_frame, cv2.COLOR_BGR2RGB)
        
        input_tensor = self.transform(frame_rgb)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        return input_tensor, (scale, x_offset, y_offset)

    def detect(self, frame):
        original_h, original_w = frame.shape[:2]
        
        input_tensor, (scale, x_offset, y_offset) = self.preprocess_frame(frame)
        
        with torch.no_grad():
            confidences, boxes = self.model(input_tensor)
        
        confidences = confidences[0].cpu().numpy()
        boxes = boxes[0].cpu().numpy()
        
        # Filter detections
        valid_detections = []
        for conf, box in zip(confidences, boxes):
            if conf > self.confidence_threshold:
                x1 = box[0] * 224 - x_offset
                y1 = box[1] * 224 - y_offset
                w = box[2] * 224
                h = box[3] * 224
                
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                w = int(w / scale)
                h = int(h / scale)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, original_w))
                y1 = max(0, min(y1, original_h))
                w = min(w, original_w - x1)
                h = min(h, original_h - y1)
                
                # Add size and aspect ratio filtering
                if self.is_valid_hand((x1, y1, w, h), original_h, original_w):
                    valid_detections.append({
                        'confidence': conf,
                        'bbox': (x1, y1, w, h)
                    })
        
        # Apply NMS to remove overlapping detections
        valid_detections = self.apply_nms(valid_detections)
        
        return valid_detections

def main():
    # Initialize detector
    model_path = 'palm_detection_checkpoints/best_model.pth'  # Adjust path as needed
    detector = HandDetector(model_path, confidence_threshold=0.5)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set up FPS calculation
    fps_history = []
    
    print("Press 'q' to quit")
    
    while True:
        start_time = time.time()
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Run detection
        detections = detector.detect(frame)
        
        # Draw detections
        for det in detections:
            x1, y1, w, h = det['bbox']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            
            # Draw confidence score
            text = f"{conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate and display FPS
        fps_history.append(1.0 / (time.time() - start_time))
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Hand Detection', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()