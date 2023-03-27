import torch, copy
from torch import nn
from torchvision.ops import FrozenBatchNorm2d
from torchvision.models import detection, video
from torchvision.transforms.functional import normalize, convert_image_dtype

from detectron2 import model_zoo
from detectron2.config import instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.backbone.utils import get_abs_pos

class ViTDetSimplePyramid(nn.Module):
    def __init__(self, weights, trainable):
        super().__init__()
        generalized_rcnn = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model
        self.mean, self.std = generalized_rcnn.pixel_mean, generalized_rcnn.pixel_std
        generalized_rcnn = instantiate(generalized_rcnn)
        DetectionCheckpointer(generalized_rcnn).resume_or_load(weights, resume=False)
        backbone = generalized_rcnn.backbone.net
        # assign
        self.pretrain_use_cls_token = backbone.pretrain_use_cls_token
        # abs pos embed
        self.pos_embed = backbone.pos_embed
        self.pos_embed.requires_grad = False
        # patch conv
        self.patch_embed = backbone.patch_embed
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # blocks
        trainable_start = len(backbone.blocks) - trainable
        self.shared_blocks = nn.Sequential(*backbone.blocks[:trainable_start])
        self.video_blocks = nn.Sequential(*backbone.blocks[trainable_start:])
        self.image_blocks = nn.Sequential(*copy.deepcopy(backbone.blocks[trainable_start:]))
        # fixed shared and video blocks
        for block in [*self.shared_blocks, *self.video_blocks]:
            block.drop_path = nn.Identity()
            for param in block.parameters():
                param.requires_grad = False
        from .memory_efficient_vitdet import _replace_vitdet_attention_forward
        _replace_vitdet_attention_forward(self.image_blocks)
    
    def forward_single(self, x, final_blocks):
        x = self.patch_embed(x)
        x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, x.shape[1:3])
        return torch.cat([final_blocks(self.shared_blocks(_x)) for _x in x.split(1)])
    
    def forward(self, images, videos):
        assert images.dtype == videos.dtype == torch.uint8
        # modified from detectron2.modeling.meta_arch.rcnn
        images = normalize(images.float(), mean=self.mean, std=self.std)
        videos = normalize(videos.float(), mean=self.mean, std=self.std)
        images = self.forward_single(images, self.image_blocks).permute(0, 3, 1, 2)
        videos = self.forward_single(videos, self.video_blocks).permute(0, 3, 1, 2)
        return images, videos

class FasterRCNNFeaturePyramid(nn.Module):
    def __init__(self, detector, weights, image_scales, video_scale):
        super().__init__()
        self.image_scales = image_scales
        self.video_scale = video_scale
        self.register_buffer('image_mean', torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,-1,1,1))
        self.register_buffer('image_std', torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,-1,1,1))
        detector = getattr(detection, detector)(weights=weights, trainable_backbone_layers=5)
        backbone = detector.backbone 
        # layer l -> p l+1, corresponding to fpn denotes
        unused_strides = []
        for layer in backbone.body.return_layers:
            stride = int(layer[-1])+1
            level = f'p{stride}'
            backbone.body.return_layers[layer] = level
            if level not in image_scales and level != video_scale:
                unused_strides.append(stride)
        for stride in unused_strides:
            backbone.fpn.layer_blocks[stride-2] = nn.Identity() 
        if 2 in unused_strides:
            backbone.body.return_layers.pop('layer1')
            backbone.fpn.inner_blocks = backbone.fpn.inner_blocks[1:]
            backbone.fpn.layer_blocks = backbone.fpn.layer_blocks[1:]
        backbone.fpn.extra_blocks = None
        
        # bn -> fbn
        for name, module in backbone.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                pointer = backbone
                nodes = name.split('.')
                for n in nodes[:-1]:
                    pointer = getattr(pointer, n)
                bn = getattr(pointer, nodes[-1])
                fbn = FrozenBatchNorm2d(bn.num_features, bn.eps)
                fbn.load_state_dict(bn.state_dict())
                setattr(pointer, nodes[-1], fbn)
        
        # perfect!
        self.backbone_with_fpn = backbone

    def preprocess(self, image):
        # from https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py
        # from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/transform.py#L148
        image = convert_image_dtype(image)
        return (image - self.image_mean) / self.image_std
    
    def forward(self, images, videos):
        assert images.dtype == videos.dtype == torch.uint8
        images = self.backbone_with_fpn(self.preprocess(images).contiguous())
        videos = self.backbone_with_fpn(self.preprocess(videos).contiguous())
        return [images[l]for l in self.image_scales], videos[self.video_scale]

if __name__ == '__main__':
    fasterrcnn_fpn = FasterRCNNFeaturePyramid(detector='fasterrcnn_resnet50_fpn_v2', weights='DEFAULT').cuda()
    print(fasterrcnn_fpn)
    images = torch.rand(2,3,256,256).to(torch.uint8).cuda()
    videos = torch.rand(24*2,3,256,256).to(torch.uint8).cuda() 
    images, videos = fasterrcnn_fpn(images,videos)
    print(images.keys(), videos.keys())
    pass