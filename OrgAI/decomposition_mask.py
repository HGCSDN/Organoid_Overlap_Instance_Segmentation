"""Overlapping-organoid mask decomposition branch for PLU."""
import torch.nn as nn

class PLU_Decomposition_Branch(nn.Module):
    """Predict candidate instance masks and existence logits."""
    def __init__(self, in_channels=256, roi_size=14, max_instances_K=5):
        super().__init__()
        self.max_K = max_instances_K
        self.mask_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.max_K, kernel_size=1),
        )
        self.count_pool = nn.AdaptiveAvgPool2d(1)
        self.count_fc = nn.Linear(256, self.max_K)

    def forward(self, roi_features):
        mask_logits = self.mask_conv(roi_features)
        count_logits = self.count_fc(self.count_pool(roi_features).flatten(1))
        return mask_logits, count_logits
