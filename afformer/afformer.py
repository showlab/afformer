import torch
from torch import nn
from torch.nn import functional as F

class Afformer(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, predictor: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        
    def forward(self, batch):
        images, videos, num_frames_list = batch[:3]
        images, videos = self.encoder(images, videos)
        heatmaps, action_logits = self.predictor(
            self.decoder(images, videos, num_frames_list)
        )
        
        if self.training:
            gt_heatmaps, gt_actions = batch[-2:]
            if gt_actions is not None:
                # only for valid heatmap loss
                heatmaps = torch.cat([heatmap[gt_action] for heatmap, gt_action in zip(heatmaps, gt_actions)])
                # view gt to loss shape
                gt_heatmaps = torch.cat(gt_heatmaps).view_as(heatmaps)
                # produce multihot label
                gt_action_logits = torch.zeros_like(action_logits, 
                    device=action_logits.device, dtype=action_logits.dtype)
                for i, gt_action in enumerate(gt_actions):
                    gt_action_logits[i].scatter_(0, gt_action, 1)
                # action loss
                loss_action = F.binary_cross_entropy_with_logits(
                    action_logits, gt_action_logits, reduction='mean')
            else:
                gt_heatmaps = gt_heatmaps.view_as(heatmaps)
                loss_action = None
            
            gt_heatmaps = gt_heatmaps.flatten(1)
            gt_heatmaps = gt_heatmaps / gt_heatmaps.sum(dim=1, keepdim=True)

            heatmaps = heatmaps.flatten(1).log_softmax(dim=1)
            loss_heatmap = F.kl_div(heatmaps, gt_heatmaps, reduction='batchmean')
            loss = loss_heatmap if loss_action is None else loss_heatmap + loss_action
            return dict(loss=loss, loss_heatmap=loss_heatmap, loss_action=loss_action)
        else:
            heatmaps = heatmaps.flatten(2).softmax(dim=-1)
            return heatmaps, action_logits.sigmoid_()
