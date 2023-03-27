import os, tqdm, torch, math
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.ops import box_area
from torchvision.transforms import Resize, RandomPerspective
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, gaussian_blur, pad

def random_include_boxes_mask(image, boxes, mask_ratio, num_masks):
    assert mask_ratio >= 1
    image = image.clone()
    h, w = image.shape[-2:]
    for box in boxes:
        delta = (mask_ratio - 1) * torch.rand(2, 2)
        # x1, y1 should - sth, x2, y2 should + sth
        delta[0] *= -1
        # x1, x2 should + w * delta, y1, y2 should + h * delta
        delta *= torch.sqrt((box[3] - box[1]) * (box[2] - box[0]))
        mask = box.view(-1,2) + delta
        # to int and clamp
        mask[:,0].clamp_(min=0, max=w-1)
        mask[:,1].clamp_(min=0, max=h-1)
        # then mask
        x1, y1, x2, y2 = mask.int().view(-1)
        image[:,y1:y2+1, x1:x2+1].random_(0,255)
        bh, bw = y2 - y1, x2 - x1
        
        for _ in range(num_masks - 1):
            # add a random box
            x1 = torch.randint(0, w-bw, size=(1,))[0]
            y1 = torch.randint(0, h-bh, size=(1,))[0]
            image[:,y1:y1+bh+1, x1:x1+bw+1].random_(0,255)
    return image

def box2pointmap(boxes, hw, dtype=torch.float, fill=1):
    # box: N x 4
    heatmap = torch.zeros(hw, dtype=dtype)
    for box in boxes.int():
        x1,y1,x2,y2 = box
        heatmap[y1:y2+1,x1:x2+1] = fill
    return heatmap

def maximum_valid_crop(image, pointmap):
    nonzero_idxs = image.sum(dim=0).nonzero()
    i1, j1 = nonzero_idxs.min(dim=0).values
    i2, j2 = nonzero_idxs.max(dim=0).values
    image = image[:,i1:i2+1,j1:j2+1]
    pointmap = pointmap[:,i1:i2+1,j1:j2+1]
    return image, pointmap

class AffoDistOPRA(Dataset):
    def __init__(self, affomined_root, mask_ratio, num_masks, distortion_scale, num_frames, clip_interval, contact_threshold) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.num_masks = num_masks
        self.distortion_scale = distortion_scale

        videos = sorted(os.listdir(affomined_root)) # avoid random
        clips_path = f'affodist/outputs/opra/clips.pth'
        self.clips = []

        if os.path.exists(clips_path):
            print(f'{clips_path} exist. load it...')
            self.clips = torch.load(clips_path)
            print('done!')
        else:
            print(f'{clips_path} not exists. make it...')
            for video in tqdm.tqdm(videos):
                video = os.path.join(affomined_root, video)
                clips = sorted(os.listdir(video)) # avoid random
                for clip in clips:
                    clip = os.path.join(video, clip)
                    clip_hands = torch.load(f'{clip}/hands.pth', map_location='cpu')
                    clip_hand_boxes, clip_contacts = zip(*[
                        self.filter(hands, contact_threshold) for hands in clip_hands
                    ])
                    self.clips.append(
                        dict(
                            root=clip,
                            hand_boxes=clip_hand_boxes,
                            contacts=clip_contacts
                        )
                    )
            print(f'making done! save to {clips_path}')
            os.makedirs(os.path.dirname(clips_path), exist_ok=True)
            torch.save(self.clips, clips_path)
            print('done!')
        
        print('extract shotcuts from all clips')
        self.cuts = [] 
        for clip in tqdm.tqdm(self.clips):
            num_total_frames = len(clip['hand_boxes'])
            assert num_total_frames >= num_frames
            last_start = len(clip['hand_boxes']) - num_frames
            starts = list(range(0, last_start+1, clip_interval))
            if starts[-1] < last_start: # add last
                starts.append(last_start)
            root = clip['root']
            self.cuts.extend([
                dict(
                    frames=[f'{root}/{i}.jpg' for i in range(s,s+num_frames)],
                    hand_boxes=clip['hand_boxes'][s:s+num_frames],
                    contacts=clip['contacts'][s:s+num_frames]
                ) for s in starts
            ])
        
        self.resize = Resize((256, 256))
        self.random_perspective = RandomPerspective(distortion_scale=distortion_scale, p=1.0) 

    def filter(self, hands, threshold):
        hand_boxes, hand_states, hand_scores = hands['hand_bboxes'], \
            hands['hand_states'], hands['hand_scores']
        contacts = hand_states == 1
        lowconf = hand_scores <= threshold
        ignore = contacts & lowconf
        select = ~ignore
        return hand_boxes[select], contacts[select]
    
    def __getitem__(self, index):
        cut = self.cuts[index]
        hand_boxes = cut['hand_boxes']
        contacts = cut['contacts']
        frames = torch.stack([
            read_image(f) for f in cut['frames']
        ])
        
        # randomly select an affordance image and make pointmap target
        frame_contacts = torch.stack([contact.any() for contact in contacts])
        frame_contact_idxs = frame_contacts.nonzero(as_tuple=True)[0]
        idx = torch.randint(len(frame_contact_idxs), (1,))[0]
        idx = frame_contact_idxs[idx]
        image = frames[idx]
        contact_boxes = hand_boxes[idx][contacts[idx]]
        pointmap = box2pointmap(contact_boxes, image.shape[-2:], dtype=torch.uint8, fill=255) # uint8 to keep cat no change image dtype
        
        # randomly mask a larger patch that include contact region
        if self.mask_ratio > 0 and self.num_masks > 0:
            image = random_include_boxes_mask(image, contact_boxes, mask_ratio=self.mask_ratio, num_masks=self.num_masks)
        
        # concatenate with image and do perspective transformation
        image, pointmap = self.random_perspective(torch.cat([image, pointmap[None]])).split([3,1])
        
        if self.distortion_scale > 0:
            image, pointmap = maximum_valid_crop(image, pointmap)
        
        frames = self.resize(frames)
        image = self.resize(image)
        heatmap = (pointmap > 0).float()
        heatmap = gaussian_blur(self.resize(heatmap), (85, 85))[0]
        
        return image, frames, len(frames), heatmap
    

    def __len__(self,):
        return len(self.cuts)

    @staticmethod
    def collate_fn(x):
        images, videos, num_frames_list, heatmaps = zip(*x)
        return torch.stack(images), torch.cat(videos), num_frames_list, None, torch.stack(heatmaps), None

from torchvision.transforms.functional import rgb_to_grayscale

def similarity(a, b):
    a, b = rgb_to_grayscale(a), rgb_to_grayscale(b)
    maximum = torch.maximum(255-a, a)
    return 1 - torch.mean(torch.sub(b,a) / torch.maximum(255-a, a))

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import to_pil_image, resize

# run from afformer dir
if __name__ == '__main__':
    mask_ratio, num_masks = 1.5, 2
    ratios = []
    affodist_dataset = AffoDistOPRA('affominer/outputs/opra', 0, 0, 0, 32, 16, 0.99)
    for i in tqdm.tqdm(range(len(affodist_dataset))):
        cut = affodist_dataset.cuts[i]
        hand_boxes = cut['hand_boxes']
        contacts = cut['contacts']

        frame_contacts = torch.stack([contact.any() for contact in contacts])
        frame_contact_idxs = frame_contacts.nonzero(as_tuple=True)[0]
        idx = torch.randint(len(frame_contact_idxs), (1,))[0]
        idx = frame_contact_idxs[idx]
        image = cut['frames'][idx]
        contact_boxes = hand_boxes[idx][contacts[idx]]
        hand_area = box_area(contact_boxes).sum()
        w, h = Image.open(image).size

        mask_area = hand_area*num_masks*mask_ratio
        total_area = w*h

        ratios.append(mask_area / total_area)