import torch
import torch.nn as nn


class PatchRealismScorer(nn.Module):
    """Lightweight patch realism scorer for knitted local texture realism.

    Input:  [B, 3, P, P]
    Output: [B] realism logits (higher = more realistic)
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 4]
        layers = []
        c_in = in_ch
        for i, c_out in enumerate(chs):
            layers.extend([
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2 if i < 3 else 1, padding=1),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            c_in = c_out
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(chs[-1], chs[-1] // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(chs[-1] // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat).squeeze(-1)
