import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math
from typing import List, Tuple, Dict
import time

# Basic building blocks
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out

class FPN(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1)
            for in_channels in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            ConvBlock(out_channels, out_channels)
            for _ in in_channels_list
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        for i in range(len(laterals)-1, 0, -1):
            laterals[i-1] += F.interpolate(laterals[i], size=laterals[i-1].shape[-2:])
        
        return [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv(x)
        return x * attention

# Main model
class ImprovedHandDetectionModel(nn.Module):
    def __init__(self, num_classes: int = 1, num_anchors: int = 9):
        super().__init__()
        self.num_anchors = num_anchors
        
        # Backbone
        self.stage1 = nn.Sequential(
            ConvBlock(3, 64, 7, 2),
            nn.MaxPool2d(3, 2, 1),
            ResBlock(64),
            ResBlock(64)
        )
        
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, stride=2),
            ResBlock(128),
            ResBlock(128)
        )
        
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, stride=2),
            ResBlock(256),
            ResBlock(256)
        )
        
        self.stage4 = nn.Sequential(
            ConvBlock(256, 512, stride=2),
            ResBlock(512),
            ResBlock(512)
        )
        
        # FPN
        self.fpn = FPN([512, 256, 128, 64], 256)
        
        # Attention modules
        self.attentions = nn.ModuleList([
            SpatialAttention(256) for _ in range(4)
        ])
        
        # Detection heads
        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                ConvBlock(256, 256),
                ConvBlock(256, 256),
                nn.Conv2d(256, num_anchors * num_classes, 1)
            ) for _ in range(4)
        ])
        
        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                ConvBlock(256, 256),
                ConvBlock(256, 256),
                nn.Conv2d(256, num_anchors * 4, 1)
            ) for _ in range(4)
        ])
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Backbone features
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        
        # FPN features
        fpn_features = self.fpn([c4, c3, c2, c1])
        
        # Apply attention
        attended_features = [att(feat) for att, feat in zip(self.attentions, fpn_features)]
        
        # Detection heads
        cls_outputs = []
        reg_outputs = []
        for feat, cls_head, reg_head in zip(attended_features, self.cls_heads, self.reg_heads):
            cls_outputs.append(cls_head(feat))
            reg_outputs.append(reg_head(feat))
        
        return cls_outputs, reg_outputs

# Loss functions
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class IoULoss(nn.Module):
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert to x1, y1, x2, y2 format
        pred_x1 = predictions[..., 0]
        pred_y1 = predictions[..., 1]
        pred_x2 = predictions[..., 0] + predictions[..., 2]
        pred_y2 = predictions[..., 1] + predictions[..., 3]
        
        target_x1 = targets[..., 0]
        target_y1 = targets[..., 1]
        target_x2 = targets[..., 0] + targets[..., 2]
        target_y2 = targets[..., 1] + targets[..., 3]
        
        # Calculate intersection area
        x1 = torch.max(pred_x1, target_x1)
        y1 = torch.max(pred_y1, target_y1)
        x2 = torch.min(pred_x2, target_x2)
        y2 = torch.min(pred_y2, target_y2)
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate union area
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union = pred_area + target_area - intersection
        
        iou = intersection / (union + 1e-6)
        return 1 - iou.mean()

class CombinedLoss(nn.Module):
    def __init__(self, cls_weight: float = 1.0, reg_weight: float = 1.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.cls_loss = FocalLoss()
        self.reg_loss = IoULoss()
    
    def forward(self, cls_preds: List[torch.Tensor], reg_preds: List[torch.Tensor],
                cls_targets: List[torch.Tensor], reg_targets: List[torch.Tensor]) -> torch.Tensor:
        cls_loss = sum(self.cls_loss(pred, target) 
                      for pred, target in zip(cls_preds, cls_targets))
        reg_loss = sum(self.reg_loss(pred, target) 
                      for pred, target in zip(reg_preds, reg_targets))
        
        return self.cls_weight * cls_loss + self.reg_weight * reg_loss

# Inference helper
class HandDetector:
    def __init__(self, model: ImprovedHandDetectionModel, device: torch.device,
                 confidence_threshold: float = 0.5, nms_threshold: float = 0.5):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        self.transform = A.Compose([
            A.Resize(320, 320),
            A.Normalize(),
            ToTensorV2(),
        ])
    
    def _process_detections(self, cls_outputs: List[torch.Tensor], 
                          reg_outputs: List[torch.Tensor],
                          original_size: Tuple[int, int]) -> List[Dict]:
        detections = []
        
        for cls_output, reg_output in zip(cls_outputs, reg_outputs):
            # Get confidence scores
            scores = torch.sigmoid(cls_output.squeeze())
            
            # Filter by confidence
            mask = scores > self.confidence_threshold
            if not mask.any():
                continue
            
            scores = scores[mask]
            boxes = reg_output.squeeze()[mask]
            
            # Convert to original image coordinates
            h, w = original_size
            scale = torch.tensor([w, h, w, h]).to(boxes.device)
            boxes = boxes * scale
            
            # Apply NMS
            keep = torchvision.ops.nms(boxes, scores, self.nms_threshold)
            
            for score, box in zip(scores[keep], boxes[keep]):
                detections.append({
                    'confidence': score.item(),
                    'bbox': box.tolist()
                })
        
        return detections
    
    @torch.no_grad()
    def detect(self, image: np.ndarray) -> List[Dict]:
        # Prepare image
        h, w = image.shape[:2]
        transformed = self.transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Run inference
        cls_outputs, reg_outputs = self.model(input_tensor)
        
        # Process detections
        detections = self._process_detections(cls_outputs, reg_outputs, (h, w))
        return detections

# Real-time detection
def run_realtime_detection():
    # Initialize model and detector
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedHandDetectionModel()
    
    # Load weights if available
    if os.path.exists('hand_detection_weights.pth'):
        model.load_state_dict(torch.load('hand_detection_weights.pth'))
    
    detector = HandDetector(model, device)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        detections = detector.detect(frame)
        
        # Draw detections
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            
            x1, y1, w, h = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {1.0/(time.time()-start_time):.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Hand Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_detection()