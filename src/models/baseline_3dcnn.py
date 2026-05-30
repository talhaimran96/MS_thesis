import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class ResNet3DBaseline(nn.Module):
    def __init__(self, num_classes=400, pretrained=True):
        super(ResNet3DBaseline, self).__init__()
        # Use a standard 3D ResNet
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        self.model = r3d_18(weights=weights)
        
        # Modify the fully connected layer for the specific number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # x is (B, C, T, H, W)
        return self.model(x)

if __name__ == "__main__":
    model = ResNet3DBaseline(num_classes=400, pretrained=False)
    x = torch.randn(2, 3, 16, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")
