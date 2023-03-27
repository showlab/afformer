from torch import nn
from detectron2.layers import get_norm
from timm.models.layers import trunc_normal_

class Predictor(nn.Module):
    def __init__(self, hidden_channels, out_channels, classify_action) -> None:
        super().__init__()
        self.classify_action = classify_action
        # following vitdet
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=2, stride=2),
            get_norm('LN', hidden_channels // 2),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_channels // 2, out_channels, kernel_size=2, stride=2),
        ) 
        if classify_action:
            # following vit with average pooling
            # self.actioner = nn.Sequential(
            #     nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
            #     get_norm('LN', hidden_channels),
            #     nn.GELU(),
            #     nn.AdaptiveAvgPool2d((1,1)),
            #     nn.Flatten(1),
            #     nn.Linear(hidden_channels, out_channels),
            # )
            self.actioner = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1, bias=False),
                get_norm('LN', hidden_channels // 2),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(1),
                nn.Linear(hidden_channels // 2, out_channels),
            )
            # self.actioner = nn.Sequential(
            #     nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            #     get_norm('LN', hidden_channels),
            #     nn.GELU(),
            #     nn.AdaptiveAvgPool2d((1,1)),
            #     nn.Flatten(1),
            #     nn.Linear(hidden_channels, out_channels),
            # )

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        return self.upsampler(x), self.actioner(x) if self.classify_action else None
