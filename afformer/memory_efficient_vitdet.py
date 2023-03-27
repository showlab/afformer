import torch
from functools import partial

from detectron2 import model_zoo
from detectron2.config import instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.backbone.utils import get_rel_pos

from xformers.ops import memory_efficient_attention

# modified from detectron2.modeling.backbone.utils.add_decomposed_rel_pos
def get_decomposed_rel_pos(q, rel_pos_h, rel_pos_w, q_size, k_size, num_heads):
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    q = q.unflatten(1, (q_h, q_w))
    rel_h = torch.einsum("bhwnc,hkc->bnhwk", q, Rh)
    rel_w = torch.einsum("bhwnc,wkc->bnhwk", q, Rw)

    # attn: bhwk b, q_h, q_w, k_h, k_w
    rel_pos = (
        rel_h[:, :, :, :, :, None] + 
        rel_w[:, :, :, :, None, :]
    ).reshape(-1, num_heads, q_h * q_w, k_h * k_w)

    return rel_pos

def memory_efficient_module_forward(self, x):
    B, H, W, _ = x.shape
    q, k, v = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).unbind(2)

    if self.use_rel_pos:
        rel_pos = get_decomposed_rel_pos(q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W), self.num_heads)
    else:
        rel_pos = None

    x = memory_efficient_attention(q, k, v, attn_bias=rel_pos).reshape(B, H, W, -1)
    x = self.proj(x)
    return x
    
def _replace_vitdet_attention_forward(network):
    from detectron2.modeling.backbone import vit
    for module in network.modules():
        if isinstance(module, vit.Attention):
            module.forward = partial(memory_efficient_module_forward, module)

if __name__ == '__main__':
    from lightning import seed_everything
    seed_everything(42)
    generalized_rcnn = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model
    generalized_rcnn.backbone.net.drop_path_rate = 0.0 # eval mode
    generalized_rcnn = instantiate(generalized_rcnn)
    _replace_vitdet_attention_forward(generalized_rcnn)
    DetectionCheckpointer(generalized_rcnn).resume_or_load('weights/mask_rcnn_vitdet_b_coco.pkl', resume=False)
    backbone = generalized_rcnn.backbone.net.cuda()
    backbone.eval()
    
    x = torch.rand(4,3,1024,1024, dtype=torch.float16, device="cuda")
    # y = backbone(x)['last_feat']
    # y_e = torch.load('y.pt')
    # print(torch.any(y != y_e))
    # pass
    
    import time
    s = time.time()
    for _ in range(10):
        with torch.no_grad():
            with torch.autocast("cuda"):
                backbone(x)['last_feat']
    e = time.time()
    print((e-s)/10, torch.cuda.max_memory_allocated()/1024**3)

    