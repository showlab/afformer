import torch, os, json, math
import torchvision
from torchvision.transforms import RandomCrop, Resize
from torchvision.transforms import functional as F 

from torchvision.io import read_image
from torchvision.ops import box_convert
from torch.utils.data import Dataset
from .metrics import KLD, SIM, AUC_Judd

def boxes2heatmap(boxes, hw, dtype=torch.float, fill=1):
    # box: N x 4
    heatmap_list = []
    for box in boxes:
        heatmap = torch.zeros(hw, dtype=dtype)
        x1, y1, x2, y2 = box.int()
        heatmap[y1:y2+1, x1:x2+1] = 1
        heatmap_list.append(heatmap)
        # cx, cy, w, h = box_convert(box, in_fmt='xyxy', out_fmt='cxcywh').int()
        # heatmap[cy, cx] = fill
        # w = w + 1 if w % 2 == 0 else w
        # h = h + 1 if h % 2 == 0 else h
        # w = h = 85
        # heatmap_list.append(F.gaussian_blur(heatmap[None], (w, h))[0])
    return torch.stack(heatmap_list).sum(dim=0)

def plot(image_path, points, augmented_image, augmented_heatmap, alpha=0.7):
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure(figsize=(3, 1))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    h, w = augmented_heatmap.shape
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    stride = 8
    image = cv2.resize(image, (h, w))
    ax1.imshow(image)

    # # Plot the points.
    # points = torch.tensor(points).numpy() // stride
    # ax1.scatter(points[:, 0], points[:, 1], c='r')
    # ax1.set_title('points annotation\nnum_points = {}'.format(points.shape[0]))

    # augmented_image = augmented_image[0].permute(1,2,0).numpy()
    # augmented_heatmap = augmented_heatmap.numpy()
    # augmented_image = augmented_image.astype('uint8')
    # augmented_image = cv2.resize(augmented_image, (h,w))
    # Plot the heatmap.
    ax2.imshow(augmented_image)
    
    # Plot the overlay of heatmap on the target image.
    processed_heatmap = augmented_heatmap * 255 / np.max(augmented_heatmap)
    processed_heatmap = np.tile(processed_heatmap[:, :, np.newaxis], (1, 1, 3))
    processed_heatmap = processed_heatmap.astype('uint8')
    assert processed_heatmap.shape == augmented_image.shape
    overlay = cv2.addWeighted(processed_heatmap, alpha, augmented_image, 1-alpha, 0)
    ax3.imshow(overlay, interpolation='nearest')
    ax3.set_title('heatmap overlay\nalpha = {}'.format(alpha))

    fig.savefig('x_centerpoint.jpg')


class AssistQButton(Dataset):
    root = 'datasets/assistq_button'

    def __init__(self, split):
        self.annos = getattr(self, f'init_{split}')(self.root)
        self.training = split == 'train'
        # self.random_crop = RandomCrop(224)
        self.resize_256 = Resize((256, 256))
        if split != 'train':
            self.evaluate = getattr(self, f'evaluate_{split}')
            self.accumulate = getattr(self, f'accumulate_{split}')
    
    def init_train(self, root):
        # annotation_path = os.path.join(root, '/Users/dongxing/project/afformer/anno_.json')
        annotation_path = os.path.join(root, "train.json")
        return json.load(open(annotation_path))

    def init_test(self, root):
        annotation_path = os.path.join(root, "test_v2.json")
        return json.load(open(annotation_path))

    def init_save(self, root):
        return self.init_test(root)

    def augment(self, image, frames, heatmaps):
        # flip
        if self.training and torch.rand(1) < 0.5:
            image = F.hflip(image)
            frames = F.hflip(frames)
            heatmaps = F.hflip(heatmaps)
        return image, frames, heatmaps

    def to_heatmaps(self, pointmaps_list, k_ratio=3.0):
        # h, w  = pointmaps.shape
        # k = int(math.sqrt(h*w) / k_ratio)
        # if k % 2 == 0:
        #     k += 1
        # heatmaps = F.gaussian_blur(pointmaps[None], (k, k))[0]
        heatmaps = torch.zeros(pointmaps_list[0].shape)
        for i, pointmaps in enumerate(pointmaps_list):
            button_size = self.box_size_list[i]
            button_size[0] = button_size[0]+1 if button_size[0] % 2 == 0 else button_size[0]
            button_size[1] = button_size[1]+1 if button_size[1] % 2 == 0 else button_size[1]            
            heatmaps_single = F.gaussian_blur(pointmaps[None], button_size)[0]
            heatmaps = heatmaps + heatmaps_single
        return heatmaps
    
    def compress(self, pointmaps, actions):
        if actions.unique().numel() != actions.numel():
            _actions, _pointmaps = [], []
            for a in range(self.num_actions):
                action_mask = actions == a
                if action_mask.sum():
                    pointmap = pointmaps[action_mask]
                    assert pointmap.dim() == 3
                    pointmap = pointmap.sum(dim=0)
                    _pointmaps.append(pointmap)
                    _actions.append(a)
            pointmaps = torch.stack(_pointmaps)
            actions = torch.tensor(_actions)
        return pointmaps, actions
    
    def resize_box(self, boxes):
        boxes[:,(0,2)] *= self.resize_factor_w
        boxes[:,(1,3)] *= self.resize_factor_h
        return boxes
        
    def __getitem__(self, index):
        anno = self.annos[index]
        image = read_image(anno['image_path'], mode= torchvision.io.image.ImageReadMode.RGB)
        image_resized = self.resize_256(image)
        c, h, w = image.shape
        _c,_h,_w = image_resized.shape

        image = image_resized
        self.resize_factor_h = _h/h
        self.resize_factor_w = _w/w

        num_frames = len(anno['frames_path'])
        if num_frames > 64:
            idxs = torch.linspace(0, num_frames-1, 64).int()
            anno_frames = [anno['frames_path'][idx] for idx in idxs]
        else:
            anno_frames = anno['frames_path']

        frames = torch.stack([self.resize_256(read_image(_)) for _ in anno_frames])
        
        boxes = torch.tensor(anno["boxes"]).float()
        boxes = self.resize_box(boxes)
        pointmaps_list = boxes2heatmap(boxes, image.shape[-2:])
        image, frames, heatmaps = self.augment(image, frames, pointmaps_list)
        
        return image, frames, len(frames), index, heatmaps, None
    
    @staticmethod
    def collate_fn(x):
        images, videos, num_frames_list, indices, heatmaps, actions = zip(*x)
        return torch.stack(images), torch.cat(videos), num_frames_list, indices, torch.cat(heatmaps), None

    def evaluate_test(self, heatmaps, batch):
        heatmaps_gt = batch[-2]
        heatmaps_gt = heatmaps_gt.view_as(heatmaps)
        # kld, sim, auc_j, n, n_auc
        results = torch.zeros(5, device=heatmaps.device)
        results[0] = KLD(heatmaps, heatmaps_gt)
        results[1] = SIM(heatmaps, heatmaps_gt)
        for p, g in zip(heatmaps, heatmaps_gt):
            auc_judd = AUC_Judd(p, g)
            if auc_judd < 0:
                continue
            results[2] += auc_judd
            results[4] += 1
        results[3] = heatmaps_gt.shape[0]
        return results

    def accumulate_test(self, results):
        results = results.sum(dim=0)
        kld, sim = results[:2] / results[3]
        auc_judd = results[2] / results[-1]
        return dict(kld=kld, sim=sim, auc_judd=auc_judd)

    def evaluate_save(self, predictions, batch):
        heatmaps, actions = predictions
        heatmaps_gt = batch[-2]
        heatmaps = heatmaps_gt.view_as(heatmaps)
        indices = batch[-3]
        heatmaps = heatmaps.cpu()
        for i, index in enumerate(indices):
            self.annos[index]['heatmap'] = heatmaps[i].view(256,256)
            # h = self.annos[index]['heatmap']
            # h = (255 * (h / h.max())).to(torch.uint8)
            # F.to_pil_image(h).save('h.jpg')
        return None
    
    def accumulate_save(self, results):
        os.makedirs('datasets/assistq_button/predictions/', exist_ok=True)
        for anno in self.annos:
            assert 'heatmap' in anno
        torch.save(self.annos, 'datasets/assistq_button/predictions/gt_buttonmap_v2.pth')

    def __len__(self, ):
        return len(self.annos)

def plot_image(augmented_image, augmented_heatmap,boxes, alpha=0.7):
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure(figsize=(3, 1))

    h, w = augmented_heatmap.shape
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # stride = 8
    # image = cv2.resize(image, (h, w))
    # ax1.imshow(image)

    # # Plot the points.
    # points = torch.tensor(points).numpy() // stride
    # ax1.scatter(points[:, 0], points[:, 1], c='r')
    # ax1.set_title('points annotation\nnum_points = {}'.format(points.shape[0]))

        
    augmented_image = augmented_image.permute(1,2,0).numpy()

    augmented_heatmap = augmented_heatmap.numpy()
    augmented_image = augmented_image.astype('uint8')

    # augmented_image = cv2.resize(augmented_image, (h,w))
    # Plot the heatmap.
    
    # Plot the overlay of heatmap on the target image.
    processed_heatmap = augmented_heatmap * 255 / np.max(augmented_heatmap)
    processed_heatmap = np.tile(processed_heatmap[:, :, np.newaxis], (1, 1, 3))
    processed_heatmap = processed_heatmap.astype('uint8')
    assert processed_heatmap.shape == augmented_image.shape
    overlay = cv2.addWeighted(processed_heatmap, alpha, augmented_image, 1-alpha, 0)
    for bbox in boxes:
        x_range_min = int(bbox[0])
        y_range_min = int(bbox[1])
        x_range_max = int(bbox[2])
        y_range_max = int(bbox[3])
        cv2.rectangle(overlay, (x_range_min, y_range_min), (x_range_max, y_range_max), (0, 0, 255), 5)
    
    cv2.imwrite('xx.jpg', overlay)
    # plt.imshow(overlay, interpolation='nearest')

    fig.savefig('x_centerpoint.jpg')
if __name__ == '__main__':
    d = AssistQButton('train')
    # for i in range(len(d)):
    for i in range(len(d)):
        image, frames, leng, anno, heatmaps, actions = d.__getitem__(i)
        print(len(frames))
        assert len(frames) <= 64
    # from torchvision.transforms.functional import to_pil_image
    # x = image * 0.3 + 255*(heatmaps/heatmaps.max()) * 0.7
    # # to_pil_image(image).save('x.jpg')
    # to_pil_image(x.to(torch.uint8)).save('h.jpg')
        
        
