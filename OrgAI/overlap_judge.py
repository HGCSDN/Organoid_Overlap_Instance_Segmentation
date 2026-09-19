"""Overlap judgment head used by pseudo-label unmixing."""
import torch.nn as nn
import torch.nn.functional as F

class PLU_Overlap_Judge(nn.Module):
    """Predict one raw reliability logit for each ROI."""
    def __init__(self, in_channels=256, roi_size=14):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, roi_features):
        x = F.relu(self.conv1(roi_features))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x).squeeze(1)
