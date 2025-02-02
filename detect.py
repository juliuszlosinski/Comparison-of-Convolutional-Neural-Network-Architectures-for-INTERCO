import os
import yaml
from datetime import datetime
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from pathlib import Path

class BackboneWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
        # Get the channel dimensions from backbone
        self.channels = {
            'layer1': 256,   # ResNet50 layer1 output channels
            'layer2': 512,   # ResNet50 layer2 output channels
            'layer3': 1024,  # ResNet50 layer3 output channels
        }
        
        # Adaptation layers with correct channel dimensions
        self.adapt1 = nn.Conv2d(self.channels['layer1'], 256, 1)
        self.adapt2 = nn.Conv2d(self.channels['layer2'], 512, 1)
        self.adapt3 = nn.Conv2d(self.channels['layer3'], 1024, 1)
        
    def forward(self, x):
        features = []
        
        # Initial layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        # Extract features at different scales
        x1 = self.backbone.layer1(x)
        features.append(self.adapt1(x1))
        
        x2 = self.backbone.layer2(x1)
        features.append(self.adapt2(x2))
        
        x3 = self.backbone.layer3(x2)
        features.append(self.adapt3(x3))
        
        return features

def create_model_with_custom_backbone():
    """Create YOLO model with custom backbone"""
    model = YOLO('yolov8n.yaml')
    
    # Use modern way to load ResNet with weights
    backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Remove the final layers we don't need
    backbone.layer4 = None
    backbone.fc = None
    backbone.avgpool = None
    
    custom_backbone = BackboneWrapper(backbone)
    model.model.model[0] = custom_backbone
    return model

def verify_feature_maps(model):
    """Verify feature map shapes"""
    dummy_input = torch.randn(1, 3, 640, 640)
    try:
        features = model.model.model[0](dummy_input)
        print("\nFeature map shapes:")
        for i, feat in enumerate(features):
            print(f"Level {i}: {feat.shape}")
        return True
    except Exception as e:
        print(f"Error verifying feature maps: {str(e)}")
        return False

def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"runs/train/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create and verify model
        print("Creating model with custom backbone...")
        model = create_model_with_custom_backbone()
        
        if not verify_feature_maps(model):
            print("Feature map verification failed. Stopping training.")
            return
        
        # Training hyperparameters
        hyp = {
            'epochs': 1,
            'batch_size': 16,
            'imgsz': 128,
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'workers': 8,
            'save_period': 10,
            'project': str(output_dir),
            'name': 'custom_backbone',
            'exist_ok': True,
            'pretrained': False,
            'optimizer': 'Adam',
            'lr0': 0.001,
            'weight_decay': 0.0005,
        }
        
        # Start training
        print("\nStarting training...")
        results = model.train(
            data='yolo_data.yaml',  # Replace with your dataset yaml
            **hyp
        )
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")

if __name__ == "__main__":
    main()