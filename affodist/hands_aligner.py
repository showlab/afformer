import torch
from torchvision.ops import box_iou, box_convert

def min2d(matrix):
    h, w = matrix.shape
    v, idx = matrix.view(-1).min(dim=0)
    return v, torch.div(idx, w, rounding_mode='trunc'), idx % w

def max2d(matrix):
    h, w = matrix.shape
    v, idx = matrix.view(-1).max(dim=0)
    return v, torch.div(idx, w, rounding_mode='trunc'), idx % w

class HandsAligner:
    def __init__(self, max_num_hands=2):
        self.all_hand_idxs = list(range(max_num_hands))
    
    def __call__(self, hand_bboxes, hand_bboxes_pre, hand_idxs_pre):
        num_hands, num_hands_pre = len(hand_bboxes), len(hand_bboxes_pre)
        if not num_hands:
            return []
        if not num_hands_pre:
            return self.all_hand_idxs[:num_hands]
        if num_hands != num_hands_pre: # 2,1 or 1,2
            similarity = box_iou(hand_bboxes, hand_bboxes_pre)
            if similarity.max() == 0:
                c = box_convert(hand_bboxes, 'xyxy', 'cxcywh')[:,:2]
                c_pre = box_convert(hand_bboxes_pre, 'xyxy', 'cxcywh')[:,:2]
                similarity = -torch.cdist(c, c_pre)
            _, i, j = max2d(similarity)
            if num_hands == 1: # 1,2
                return [hand_idxs_pre[j]]
            else: # 2,1
                return self.all_hand_idxs if i == hand_idxs_pre[j] else self.all_hand_idxs[::-1]
        else: # 1,1 or 2,2
            if len(hand_bboxes) == 2:
                # direction similarity
                c = box_convert(hand_bboxes, 'xyxy', 'cxcywh')[:,:2]
                c_pre = box_convert(hand_bboxes_pre, 'xyxy', 'cxcywh')[:,:2]
                direction_similarity = torch.cosine_similarity(
                    (c[0] - c[1])[None], (c_pre[0] - c_pre[1])[None])

                # iou similarity
                iou_similarity = box_iou(hand_bboxes, hand_bboxes_pre)
                if iou_similarity.max() == 0:
                    cdists = torch.cdist(c, c_pre)
                    iou_similarity = 1 - cdists / cdists.sum(dim=1, keepdim=True)
                u, i, j = max2d(iou_similarity)
                iou_similarity = u / (iou_similarity[i].sum() + 1e-8)
                if i != j:
                    iou_similarity = -iou_similarity
                similarity = iou_similarity * 0.25 + direction_similarity
                if similarity < 0:
                    return hand_idxs_pre[::-1]
            return hand_idxs_pre
