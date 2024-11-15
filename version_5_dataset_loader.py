import os
import json
import urllib.request
import zipfile
import tarfile
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

class HandDatasetDownloader:
    def __init__(self, base_dir="hand_datasets"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
    def download_file(self, url, filename):
        """Download file with progress bar"""
        filepath = os.path.join(self.base_dir, filename)
        if os.path.exists(filepath):
            print(f"File already exists: {filepath}")
            return filepath
            
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, filepath)
        return filepath

    def download_egohands(self):
        """
        Download and process EgoHands dataset
        http://vision.soic.indiana.edu/projects/egohands/
        """
        dataset_dir = os.path.join(self.base_dir, "egohands")
        if os.path.exists(dataset_dir):
            print("EgoHands dataset already downloaded")
            return dataset_dir
            
        # Download dataset
        url = "http://vision.soic.indiana.edu/egohands_files/egohands_data.zip"
        zip_path = self.download_file(url, "egohands_data.zip")
        
        # Extract dataset
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        
        # Process annotations
        annotations = {}
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith("_bbox.txt"):
                    img_file = file.replace("_bbox.txt", ".jpg")
                    img_path = os.path.join(root, img_file)
                    if not os.path.exists(img_path):
                        continue
                        
                    # Read bounding boxes
                    bbox_path = os.path.join(root, file)
                    boxes = []
                    with open(bbox_path, 'r') as f:
                        for line in f:
                            x1, y1, x2, y2 = map(float, line.strip().split())
                            # Convert to normalized coordinates
                            img = cv2.imread(img_path)
                            h, w = img.shape[:2]
                            boxes.append([
                                x1/w, y1/h,  # normalized x, y
                                (x2-x1)/w, (y2-y1)/h  # normalized width, height
                            ])
                    
                    if boxes:  # Only add if hands are present
                        annotations[img_file] = {
                            "boxes": boxes,
                            "source": "egohands"
                        }
        
        # Save processed annotations
        with open(os.path.join(dataset_dir, "annotations.json"), 'w') as f:
            json.dump(annotations, f)
            
        return dataset_dir

    def download_oxford_hands(self):
        """
        Download and process Oxford Hands dataset
        https://www.robots.ox.ac.uk/~vgg/data/hands/
        """
        dataset_dir = os.path.join(self.base_dir, "oxford_hands")
        if os.path.exists(dataset_dir):
            print("Oxford Hands dataset already downloaded")
            return dataset_dir
            
        # Download dataset
        url = "https://www.robots.ox.ac.uk/~vgg/data/hands/downloads/hand_dataset.tar.gz"
        tar_path = self.download_file(url, "oxford_hands.tar.gz")
        
        # Extract dataset
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            tar_ref.extractall(dataset_dir)
        
        # Process annotations
        annotations = {}
        annotation_file = os.path.join(dataset_dir, "hand_dataset", "training_dataset", "training_data.txt")
        
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                img_file = parts[0]
                img_path = os.path.join(dataset_dir, "hand_dataset", img_file)
                if not os.path.exists(img_path):
                    continue
                
                # Get image dimensions
                img = cv2.imread(img_path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                
                # Parse boxes (format: [x1 y1 x2 y2])
                boxes = []
                num_boxes = int(parts[1])
                for i in range(num_boxes):
                    if len(parts) < 6 + i*4:
                        break
                    x1, y1, x2, y2 = map(float, parts[2+i*4:6+i*4])
                    # Convert to normalized coordinates
                    boxes.append([
                        x1/w, y1/h,  # normalized x, y
                        (x2-x1)/w, (y2-y1)/h  # normalized width, height
                    ])
                
                if boxes:  # Only add if hands are present
                    annotations[img_file] = {
                        "boxes": boxes,
                        "source": "oxford"
                    }
        
        # Save processed annotations
        with open(os.path.join(dataset_dir, "annotations.json"), 'w') as f:
            json.dump(annotations, f)
            
        return dataset_dir

class CombinedHandDataset(Dataset):
    def __init__(self, dataset_dirs, transform=None):
        self.transform = transform
        self.annotations = {}
        self.image_files = []
        
        # Load annotations from all datasets
        for dataset_dir in dataset_dirs:
            ann_file = os.path.join(dataset_dir, "annotations.json")
            if not os.path.exists(ann_file):
                continue
                
            with open(ann_file, 'r') as f:
                dataset_anns = json.load(f)
                
            # Add full path to image files
            for img_file, ann in dataset_anns.items():
                full_path = os.path.join(dataset_dir, img_file)
                if os.path.exists(full_path):
                    self.annotations[full_path] = ann
                    self.image_files.append(full_path)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        ann = self.annotations[img_path]
        boxes = torch.tensor(ann["boxes"], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, boxes

def create_combined_dataset(base_dir="hand_datasets"):
    """Download and combine multiple hand datasets"""
    # Initialize downloader
    downloader = HandDatasetDownloader(base_dir)
    
    # Download datasets
    dataset_dirs = [
        downloader.download_egohands(),
        downloader.download_oxford_hands()
    ]
    
    # Create transform pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create combined dataset
    dataset = CombinedHandDataset(dataset_dirs, transform=transform)
    
    print(f"\nCombined dataset created with {len(dataset)} images")
    return dataset

if __name__ == "__main__":
    dataset = create_combined_dataset()