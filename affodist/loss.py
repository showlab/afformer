import torch
from torch.functional import F

def b2h_distance(boxes, heatmap, norm_term): 
    # NOTE: heatmap should be normalized to 1 sum
    # heatmap = heatmap.view(1,-1).softmax(dim=1).view_as(heatmap)
    
    ij = heatmap.nonzero()
    h = heatmap[ij[:,0], ij[:,1]]
    
    # 1. if the point is in 1 bbox at least, then ignore it
    i1j1, i2j2 = boxes[:,(1,0)], boxes[:,(3,2)]
    # greater than topleft
    gt_tl_and_le_br = (i1j1[:,None] <= ij[None]) & (i2j2[:,None] >= ij[None])
    # at least in 1 box
    inbox = torch.any(gt_tl_and_le_br.all(dim=-1), dim=0)
    outbox = ~inbox
    ij, h = ij[outbox], h[outbox]
    gt_tl_and_le_br = gt_tl_and_le_br[:,outbox]

    # 2. compute the shortest distance between these point to the bboxes
    # NOTE: we use L1 distance as it is easy to implement
    tl_br = boxes[:,(1,0,3,2)].view(-1,2,2) # topleft + bottomright
    tr_bl = boxes[:,(1,2,3,0)].view(-1,2,2) # topright + buttonleft
    vertices = torch.cat([tl_br, tr_bl], dim=1)

    dist = torch.abs(vertices[:,None] - ij[None,:,None,:])

    # NOTE this is the distance to vertex, we should ignore some parallel parts
    outrange = ~gt_tl_and_le_br
    dist = dist * outrange[:,:,None]

    # affo points <-> all hand boxes distance
    dist = dist.pow(2).sum(dim=-1).min(dim=-1).values

    return torch.sum(h * dist, dim=1)  / norm_term

def pairwise_h2h_distance(heatmaps):
    # T, H, W
    N = heatmaps.shape[0]
    heatmaps_log = heatmaps.log()
    distance = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                distance = distance + F.kl_div(
                    heatmaps_log[i], heatmaps[j], reduction='sum'
                )
    return distance

class AffoDistLoss(torch.nn.Module):
    def __init__(self, w_affo2hand, w_affo2affo, uncontact_margin):
        super().__init__()
        self.w_affo2hand = w_affo2hand
        self.w_affo2affo = w_affo2affo
        self.uncontact_margin = uncontact_margin

    def affo2hand_dist2loss(self, distances, hand_contacts, margin=1):
        # 0: no contact, 1: contact
        loss_contact = distances[hand_contacts].sum()
        loss_uncontact = torch.clamp(margin - distances[~hand_contacts], min=0).sum()
        return loss_contact + loss_uncontact
    
    def forward(self, heatmaps, hand_boxes, hand_contacts, hand_heatmaps):
        N, T, H, W = heatmaps.shape
        norm_term = H**2 + W**2
        uncontact_margin = self.uncontact_margin / norm_term

        # loss_affo2affo = 0
        # for heatmaps_b in heatmaps:
        #     loss_affo2affo = loss_affo2affo + pairwise_h2h_distance(heatmaps_b) 

        # loss_affo2affo = self.w_affo2affo * loss_affo2affo / (N*H*W)

        # investigate some formats
        loss_affo2hand = 0

        contacts, uncontacts = [], []

        for heatmaps_b, hand_contacts_b, hand_heatmaps_b in zip(heatmaps, hand_contacts, hand_heatmaps):
            for heatmap_bt, hand_contacts_bt, hand_heatmap_bt in zip(heatmaps_b, hand_contacts_b, hand_heatmaps_b):
                # contact heatmap
                contact_hand_heatmap_bt = hand_heatmap_bt[hand_contacts_bt]
                uncontact_hand_heatmap_bt = hand_heatmap_bt[~hand_contacts_bt]
                if contact_hand_heatmap_bt.numel() > 0:
                    contact_mask = contact_hand_heatmap_bt.any(dim=0)
                    contacts.append(heatmap_bt[contact_mask])
                    uncontacts.append(heatmap_bt[~contact_mask])
                if uncontact_hand_heatmap_bt.numel() > 0:
                    uncontact_mask = uncontact_hand_heatmap_bt.any(dim=0)
                    uncontacts.append(heatmap_bt[uncontact_mask])

        contacts = torch.cat(contacts)
        uncontacts = torch.cat(uncontacts)

        contacts_gt = torch.ones_like(contacts, device=contacts.device)
        uncontacts_gt = torch.zeros_like(uncontacts, device=uncontacts.device)

        contact_loss = F.binary_cross_entropy_with_logits(contacts, contacts_gt, reduction='mean')
        uncontact_loss = F.binary_cross_entropy_with_logits(uncontacts, uncontacts_gt, reduction='mean')
        
        loss = contact_loss + uncontact_loss
        # for heatmaps_b, hand_boxes_b, hand_states_b in zip(heatmaps, hand_boxes, hand_states):
        #     for heatmap, hand_boxes_bt, hand_states_bt in zip(heatmaps_b, hand_boxes_b, hand_states_b):
        #         distance = b2h_distance(hand_boxes_bt, heatmap, norm_term)
        #         loss_affo2hand = loss_affo2hand + self.affo2hand_dist2loss(
        #             distance, hand_states_bt, uncontact_margin)

        # loss_affo2hand = loss_affo2hand / (N*T)

        return dict(loss=loss, contact_loss=contact_loss, uncontact_loss=uncontact_loss)

if __name__ == '__main__':
    heatmap = torch.zeros(10, 10, dtype=torch.float)
    heatmap[0,0] = 1
    boxes = torch.tensor(
        [[9.999,9.999,9.999,9.999]]
    )
    print(b2h_distance(boxes, heatmap))