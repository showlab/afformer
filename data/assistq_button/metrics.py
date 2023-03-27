import torch
from torch.functional import F

# KL Divergence
# kld(map2||map1) -- map2 is gt
def KLD(map1, map2, eps = 1e-12): # map1 has been softmaxed
    assert map1.dim() == map2.dim() == 2
    map1, map2 = map1/map1.sum(dim=1, keepdim=True), map2/map2.sum(dim=1, keepdim=True)
    return F.kl_div(map1.log(), map2, reduction='sum')

# historgram intersection
def SIM(map1, map2, eps=1e-12):
    assert map1.dim() == map2.dim() == 2
    map1, map2 = map1/map1.sum(dim=1, keepdim=True), map2/map2.sum(dim=1, keepdim=True)
    return torch.minimum(map1, map2).sum()

# AUC-J
def AUC_Judd(saliency_map, fixation_map, eps=1e-12):
    assert saliency_map.dim() == fixation_map.dim() == 1
    fixation_map = fixation_map / fixation_map.max()
    fixation_map = fixation_map > 0.5
    if not fixation_map.any():
        return -1

    # Normalize saliency map to have values between [0,1]
    saliency_map =  (saliency_map - saliency_map.min()) / (
        saliency_map.max() - saliency_map.min()
    )
    
    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)

    # Calculate AUC
    thresholds = torch.sort(S_fix, descending=True)[0]
    tp = torch.zeros(len(thresholds)+2, device=thresholds.device)
    fp = torch.zeros(len(thresholds)+2, device=thresholds.device)
    tp[0] = 0; tp[-1] = 1
    fp[0] = 0; fp[-1] = 1
    k = torch.arange(1, len(thresholds)+1, device=thresholds.device)
    above_th = torch.sum(S >= thresholds[:,None], dim=1)
    tp[1:-1] = k / n_fix 
    fp[1:-1] = (above_th - k) / (n_pixels - n_fix)
    return torch.trapz(tp, fp)