import torch
from torch import nn

from .attention import Cross3DBlock, Self2DCross3DBlock

class LayerNorm3D(nn.Module):
    def __init__(self, normalized_shape, reshape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape).reshape(reshape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape).reshape(reshape))
        self.eps = eps
        self.dim = [dim for dim, r in enumerate(reshape) if r == -1][0]
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(self.dim, keepdim=True)
        s = (x - u).pow(2).mean(self.dim, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x

class SimpleDecoder(nn.Module):
    def __init__(self, hidden_channels, drop_path, q_thw, kv_thw):
        super().__init__()
        self.q_thw = q_thw
        self.kv_thw = kv_thw
        self.hidden_channels = hidden_channels
        self.v3d = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.cross_block_3d = Cross3DBlock(
            hidden_channels, num_heads=hidden_channels//64,
            drop_path=drop_path, q_thw=q_thw, kv_thw=kv_thw,
        )
    
    def forward(self, images, videos, num_frames_list):
        videos = self.v3d(videos).transpose(0,1).split(num_frames_list, dim=1) 
        decodings = torch.cat([
            self.cross_block_3d(
                image.flatten(1).transpose(0,1).unsqueeze(0), 
                video.flatten(1).transpose(0,1).unsqueeze(0), 
                q_thw=self.q_thw, kv_thw=video.shape[1:]
            ) for image, video in zip(images, videos)
        ]).transpose(1,2).reshape_as(images).contiguous()
        return decodings
        
class FinegrainedDecoder(nn.Module):
    def __init__(self, hidden_channels, drop_path, max_q_thw, max_kv_thw):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.max_q_thw = max_q_thw
        self.max_kv_thw = max_kv_thw
        
        # temporal downsampling
        self.v3ds = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, 
                kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0), bias=False),
            LayerNorm3D(hidden_channels, (-1,1,1,1)),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, 
                kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0), bias=False),
        )
        
        self.image_level_embeds = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_channels)) for _ in range(3)]
        )
        self.self2d_cross3d_block = Self2DCross3DBlock(
            hidden_channels, num_heads=hidden_channels//64,
            drop_path=drop_path, max_q_thw=max_q_thw, max_kv_thw=max_kv_thw
        )
    
    def get_temporal_pyramids(self, videos):
        videos_t2 = [self.v3ds[0](video) for video in videos]
        videos_t4 = [self.v3ds[1:](video_t2) for video_t2 in videos_t2]
        return [videos, videos_t2, videos_t4]
    
    def forward(self, images_list, videos, num_frames_list):
        videos = videos.transpose(0,1).split(num_frames_list, dim=1) 
        videos_list = self.get_temporal_pyramids(videos)
        decodings = None
        for l, (images, videos) in enumerate(zip(images_list, videos_list)):
            # add last decodings
            images = images if decodings is None else \
                images + nn.functional.interpolate(decodings, scale_factor=2)  
            # decoder layer
            decodings = torch.cat([
                self.self2d_cross3d_block(
                    image.flatten(1).transpose(0,1).unsqueeze(0) + self.image_level_embeds[l], 
                    video.flatten(1).transpose(0,1).unsqueeze(0),
                    q_thw=(1,*image.shape[-2:]), kv_thw=video.shape[1:]
                ) for image, video in zip(images, videos)
            ]).transpose(1,2).reshape_as(images).contiguous()
        return decodings