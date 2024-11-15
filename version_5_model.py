import os
import urllib.request
import zipfile
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy.io import loadmat
from tqdm import tqdm
from glob import glob
from datetime import datetime

class EgoHandsDataset(Dataset):
    def __init__(self, root_dir, transform=None, min_confidence=0.5):
        self.root_dir = root_dir
        self.transform = transform
        self.min_confidence = min_confidence
        self.samples = []
        
        # Download dataset if not exists
        if not os.path.exists(root_dir):
            self._download_dataset()
        
        # Process all .mat files
        self._process_annotations()
    
    def _download_dataset(self):
        """Download and extract EgoHands dataset"""
        os.makedirs(self.root_dir, exist_ok=True)
        
        # Download
        url = "http://vision.soic.indiana.edu/egohands_files/egohands_data.zip"
        zip_path = os.path.join(self.root_dir, "egohands_data.zip")
        print(f"Downloading EgoHands dataset from {url}")
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root_dir)
        
        # Remove zip file
        os.remove(zip_path)
    
    def _process_annotations(self):
        """Process all .mat files in the dataset"""
        print("Processing EgoHands annotations...")
        
        # Find all directories containing data
        metadata_path = os.path.join(self.root_dir, "_LABELLED_SAMPLES")
        data_dirs = [d for d in glob(os.path.join(metadata_path, "*")) if os.path.isdir(d)]
        
        # Process each directory
        for data_dir in tqdm(data_dirs, desc="Processing directories"):
            # Load the metadata
            mat_path = os.path.join(data_dir, "polygons.mat")
            if not os.path.exists(mat_path):
                continue
                
            try:
                mat_data = loadmat(mat_path)
                polygons_data = mat_data['polygons'][0]  # Get the polygons data
                
                # Get list of frame files
                frame_files = sorted(glob(os.path.join(data_dir, "frame_*.jpg")))
                
                # Process each frame that has corresponding annotation
                for frame_idx, frame_path in enumerate(frame_files):
                    if frame_idx >= len(polygons_data):
                        break
                        
                    # Read image to get dimensions
                    img = cv2.imread(frame_path)
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    
                    boxes = []
                    # Process each hand annotation in the frame
                    for hand_polygon in polygons_data[frame_idx]:
                        # Skip if no points
                        if len(hand_polygon) == 0:
                            continue
                            
                        try:
                            # Convert polygon points to numpy array
                            points = hand_polygon.reshape(-1, 2)
                            
                            if points.size == 0:  # Skip if no points
                                continue
                            
                            # Get bounding box from polygon
                            x1 = float(np.min(points[:, 0]))
                            y1 = float(np.min(points[:, 1]))
                            x2 = float(np.max(points[:, 0]))
                            y2 = float(np.max(points[:, 1]))
                            
                            # Normalize coordinates
                            box = [
                                max(0.0, x1/w),  # x1
                                max(0.0, y1/h),  # y1
                                min(1.0, (x2-x1)/w),  # width
                                min(1.0, (y2-y1)/h)   # height
                            ]
                            
                            # Add box if it's valid
                            if box[2] > 0 and box[3] > 0:
                                boxes.append(box)
                                
                        except (IndexError, ValueError) as e:
                            print(f"Error processing hand in {frame_path}: {e}")
                            continue
                    
                    # Add sample if it has valid boxes
                    if boxes:
                        self.samples.append({
                            'image_path': frame_path,
                            'boxes': boxes,
                            'source': os.path.relpath(frame_path, self.root_dir)
                        })
                        
            except Exception as e:
                print(f"Error processing directory {data_dir}: {e}")
                continue
        
        print(f"Processed {len(self.samples)} valid frames with hand annotations")
        
        # Verify data
        if not self.samples:
            raise ValueError("No valid samples found in the dataset")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get boxes
        boxes = torch.tensor(sample['boxes'], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, boxes

class PalmDetectionModel(nn.Module):
    def __init__(self, max_boxes=4):  # Reduced max_boxes to match typical hand count
        super(PalmDetectionModel, self).__init__()
        self.max_boxes = max_boxes
        
        # Feature extraction backbone (using ResNet-like architecture)
        self.features = nn.Sequential(
            # Initial conv block
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Residual blocks
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
            self._make_layer(256, 512, 2),
            
            # Final layers
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Regression head for bounding boxes
        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, max_boxes * 5)  # 5 values per box (4 coords + 1 conf)
        )
    
    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            # Conv block with skip connection
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.bbox_head(features)
        
        # Reshape output to separate boxes
        batch_size = output.shape[0]
        output = output.view(batch_size, self.max_boxes, 5)
        
        # Split into confidence and bbox predictions
        confidence = torch.sigmoid(output[..., 0])  # [batch_size, max_boxes]
        bbox = output[..., 1:]  # [batch_size, max_boxes, 4]
        
        return confidence, bbox

class PalmDetectionLoss(nn.Module):
    def __init__(self, bbox_weight=1.0, conf_weight=1.0):
        super(PalmDetectionLoss, self).__init__()
        self.bbox_weight = bbox_weight
        self.conf_weight = conf_weight
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
    
    def forward(self, pred_conf, pred_bbox, target_bbox):
        batch_size = target_bbox.size(0)
        max_boxes = target_bbox.size(1)
        
        # Create mask for valid boxes (non-padded)
        valid_mask = (target_bbox.sum(dim=2) != 0)  # Shape: [batch_size, max_boxes]
        
        # Ensure pred_conf matches the shape of valid_mask
        pred_conf = pred_conf.view(batch_size, max_boxes)
        target_conf = valid_mask.float()
        
        # Confidence loss
        conf_loss = self.bce_loss(pred_conf, target_conf)
        
        # Box loss (only for valid boxes)
        if valid_mask.any():
            # Reshape predictions and targets for valid boxes only
            valid_pred_bbox = pred_bbox[valid_mask]
            valid_target_bbox = target_bbox[valid_mask]
            bbox_loss = self.smooth_l1_loss(valid_pred_bbox, valid_target_bbox)
        else:
            bbox_loss = torch.tensor(0.0).to(pred_conf.device)
        
        total_loss = self.conf_weight * conf_loss + self.bbox_weight * bbox_loss
        return total_loss

class PalmDetectionTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=0.001,
        checkpoint_dir='checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize training components
        self.criterion = PalmDetectionLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for images, target_boxes in progress_bar:
            images = images.to(self.device)
            target_boxes = target_boxes.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_conf, pred_bbox = self.model(images)
            
            # Compute loss
            loss = self.criterion(pred_conf, pred_bbox, target_boxes)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update stats
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        for images, target_boxes in self.val_loader:
            images = images.to(self.device)
            target_boxes = target_boxes.to(self.device)
            
            # Forward pass
            pred_conf, pred_bbox = self.model(images)
            
            # Compute loss
            loss = self.criterion(pred_conf, pred_bbox, target_boxes)
            total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            # Training phase
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation phase
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        # Plot training history
        self.plot_training_history()
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f'Checkpoint saved: {path}')
    
    def load_checkpoint(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f'Checkpoint loaded: {path}')
    
    def plot_training_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(self.checkpoint_dir, 'training_history.png')
        plt.savefig(plot_path)
        plt.close()

def create_egohands_dataset(root_dir="datasets/egohands"):
    """Create EgoHands dataset with standard transformations"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    dataset = EgoHandsDataset(
        root_dir=root_dir,
        transform=transform
    )
    
    return dataset

def collate_fn(batch):
    """Custom collate function to handle variable number of boxes with fixed max_boxes"""
    MAX_BOXES = 4  # Match with model's max_boxes
    
    images = []
    boxes = []
    
    for image, box in batch:
        images.append(image)
        
        # Ensure we only take up to MAX_BOXES boxes
        if box.size(0) > MAX_BOXES:
            box = box[:MAX_BOXES]
        
        # Pad if we have fewer than MAX_BOXES
        num_boxes = box.size(0)
        if num_boxes < MAX_BOXES:
            padding = torch.zeros((MAX_BOXES - num_boxes, 4), dtype=box.dtype)
            box = torch.cat([box, padding], dim=0)
        
        boxes.append(box)
    
    images = torch.stack(images, 0)
    boxes = torch.stack(boxes, 0)
    
    return images, boxes

def create_data_loaders(dataset, batch_size=32, train_split=0.8):
    """Create train and validation data loaders with proper collate_fn"""
    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def visualize_sample(dataset, idx):
    """Utility function to visualize a sample from the dataset"""
    image, boxes = dataset[idx]
    
    # Convert tensor to numpy if necessary
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
        # Denormalize if using standard normalization
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = (image * 255).astype(np.uint8)
    
    # Convert boxes to pixel coordinates
    h, w = image.shape[:2]
    boxes = boxes.numpy()
    
    # Draw boxes
    image_with_boxes = image.copy()
    for box in boxes:
        # Skip zero boxes (padding)
        if box.sum() == 0:
            continue
            
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int((box[0] + box[2]) * w)
        y2 = int((box[1] + box[3]) * h)
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image_with_boxes)
    plt.axis('off')
    plt.show()

def inference(model, image_path, device, confidence_threshold=0.5):
    """Run inference on a single image"""
    model.eval()
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Transform and add batch dimension
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        confidences, boxes = model(image_tensor)
        
    # Convert to numpy and filter by confidence
    confidences = confidences[0].cpu().numpy()
    boxes = boxes[0].cpu().numpy()
    
    valid_detections = []
    for conf, box in zip(confidences, boxes):
        if conf > confidence_threshold:
            valid_detections.append({
                'confidence': conf,
                'box': box
            })
    
    return valid_detections, original_size

def draw_detections(image_path, detections, original_size):
    """Draw detection boxes on the image"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = original_size
    
    for det in detections:
        box = det['box']
        conf = det['confidence']
        
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int((box[0] + box[2]) * w)
        y2 = int((box[1] + box[3]) * h)
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw confidence
        cv2.putText(image, f'{conf:.2f}', (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

if __name__ == "__main__":
    # Create dataset and loaders
    dataset = create_egohands_dataset()
    train_loader, val_loader = create_data_loaders(dataset, batch_size=32)
    
    # Initialize model and trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PalmDetectionModel(max_boxes=4)  # Match with collate_fn MAX_BOXES
    
    trainer = PalmDetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001,
        checkpoint_dir='palm_detection_checkpoints'
    )
    
    # Train model
    trainer.train(num_epochs=50)
    
    # Optional: Visualize some samples
    for i in range(5):
        visualize_sample(dataset, i)