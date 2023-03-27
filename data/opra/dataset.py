import torch, os, json, math, tqdm
from torch.utils.data import Dataset
from torchvision.transforms.functional import gaussian_blur, hflip
from torchvision.datasets.video_utils import VideoClips
from torchvision.io import read_image, ImageReadMode
from ..video_io import read_video_without_audio

# modified from torchvision.datasets.video_utils.VideoClips
class UnevenVideoClips(VideoClips):
    @staticmethod
    def _resample_video_idx(original_frames, original_fps, new_fps):
        assert original_frames > 0
        step = original_fps / new_fps
        idxs = torch.arange(0, original_frames + step, step)
        idxs = idxs[idxs <= original_frames - 1].round().long()
        return idxs
    
    @staticmethod
    def compute_clips_for_video(video_pts, max_frames, step, fps, frame_rate):
        total_frames = len(video_pts)
        _idxs = UnevenVideoClips._resample_video_idx(total_frames, fps, frame_rate)
        video_pts = video_pts[_idxs]
        if len(video_pts) <= max_frames:
            clips = [video_pts]
            idxs = [_idxs]
        else:
            clips = list(video_pts.unfold(0, max_frames, step).unbind(0))
            idxs = list(_idxs.unfold(0, max_frames, step).unbind(0))
            # need to complement clips if have remaining frames
            remaining = len(video_pts) % step
            if remaining > 0:
                clips.append(video_pts[-remaining:])
                idxs.append(_idxs[-remaining:])
        return clips, idxs
    
    def get_clip(self, idx):
        if idx >= self.num_clips():
            raise IndexError(f"Index {idx} out of range ({self.num_clips()} number of clips)")
        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]
        
        start_pts = clip_pts[0].item()
        end_pts = clip_pts[-1].item()
        video = read_video_without_audio(video_path, start_pts, end_pts, output_format=self.output_format)
        
        resampling_idx = self.resampling_idxs[video_idx][clip_idx]
        resampling_idx = resampling_idx - resampling_idx[0]
        video = video[resampling_idx[resampling_idx < len(video)]]
        
        return video, video_idx

class OPRA(Dataset):
    root = 'datasets/opra'
    num_actions = 7
    actions = ['hold', 'touch', 'rotate', 'push', 'pull', 'pick up', 'put down']

    def __init__(self, split, clip_length_in_frames, frames_between_clips, frame_rate, resize):
        super().__init__()
        self.resize = resize
        self.training = split == 'train'
        annotation_path = os.path.join(self.root, f'annotations/{split}.json')
        self.annos = json.load(open(annotation_path))
        
        if not self.training:
            self.collect = getattr(self, f'collect_{split}')
            self.accumulate = getattr(self, f'accumulate_{split}')
            self.evaluate = getattr(self, f'evaluate_{split}')
            
            # load evaluation ground-truth
            eval_gt_heatmaps_path = os.path.join(self.root, f'annotations/{split}_gt_heatmaps.pt')
            eval_gt_actions_path = os.path.join(self.root, f'annotations/{split}_gt_actions.pt')
            if os.path.exists(eval_gt_heatmaps_path):
                self.eval_gt_heatmaps = torch.load(eval_gt_heatmaps_path)
                self.eval_gt_actions = torch.load(eval_gt_actions_path)
                print(f'load cached {split}_gt_heatmaps.pt and {split}_gt_actions.pt ...')
            else:
                self.eval_gt_heatmaps = {}
                self.eval_gt_actions = {}
                print(f'compute and cache {split}_gt_heatmaps.pt and {split}_gt_actions.pt ...')
                for video_idx, anno in tqdm.tqdm(enumerate(self.annos)):
                    heatmap, action = self.make_gt(anno)
                    self.eval_gt_heatmaps[video_idx] = heatmap
                    self.eval_gt_actions[video_idx] = action
                if torch.cuda.current_device() == 0: # avoid conflicting
                    torch.save(self.eval_gt_heatmaps, eval_gt_heatmaps_path)
                    torch.save(self.eval_gt_actions, eval_gt_actions_path)

        video_paths = [anno['video_path'] for anno in self.annos]
        meta_path = os.path.join(self.root, f'annotations/{split}_meta.pt')
        if os.path.exists(meta_path):
            self.video_clips = UnevenVideoClips(video_paths, 
                clip_length_in_frames=clip_length_in_frames, 
                frames_between_clips=frames_between_clips,
                frame_rate=frame_rate,
                _precomputed_metadata=torch.load(meta_path), 
                num_workers=4, output_format='TCHW')
            print('load cached _precomputed_metadata')
        else:
            self.video_clips = UnevenVideoClips(video_paths, 
                clip_length_in_frames=clip_length_in_frames, 
                frames_between_clips=frames_between_clips,
                frame_rate=frame_rate, num_workers=4, 
                output_format='TCHW')
            if torch.cuda.current_device() == 0: # avoid conflicting
                torch.save(self.video_clips.metadata, meta_path)
    
    @staticmethod
    def to_heatmaps(pointmaps, k_ratio=3.0, offset=1e-6):
        c, h, w  = pointmaps.shape
        k = int(math.sqrt(h*w) / k_ratio)
        if k % 2 == 0:
            k += 1
        # offset trick to avoid kld nan
        heatmaps = gaussian_blur(pointmaps + offset, (k, k))
        return heatmaps
    
    @staticmethod
    def make_gt(anno):
        actions = torch.tensor(anno['actions']) - 1 # opra
        pointmaps = torch.zeros(OPRA.num_actions, anno['height'], anno['width'])
        for points, action in zip(anno['heatmaps'], actions):
            x, y = torch.tensor(points).long().hsplit(2)
            pointmaps[action, y, x] = 1
        actions = actions.unique(sorted=False)
        pointmaps = pointmaps[actions]
        heatmaps = OPRA.to_heatmaps(pointmaps)
        return heatmaps, actions
    
    def __getitem__(self, idx):
        frames, video_idx = self.video_clips.get_clip(idx)
        anno = self.annos[video_idx]
        image = read_image(anno['image_path'], ImageReadMode.RGB)
        image = self.resize(image)
        frames = self.resize(frames)
        if self.training:
            heatmaps, actions = OPRA.make_gt(anno)
            if torch.rand(1) < 0.5:
                image, frames, heatmaps = hflip(image), hflip(frames), hflip(heatmaps)
            return image, frames, len(frames), video_idx, heatmaps, actions
        else:
            return image, frames, len(frames), video_idx, None, None
    
    @staticmethod
    def collate_fn(x):
        images, videos, num_frames_list, indices, heatmaps, actions = zip(*x)
        return torch.stack(images), torch.cat(videos), num_frames_list, indices, heatmaps, actions

    def collect_test(self, predictions, batch):
        heatmaps, actions = predictions
        video_idxs = batch[-3]
        heatmaps = heatmaps.cpu()
        actions = actions.cpu()
        predictions = [{
            'video_idx': video_idx,
            'heatmap': heatmap[self.eval_gt_actions[video_idx]], 
            'action': action,
        } for video_idx, heatmap, action in zip(video_idxs, heatmaps, actions)]
        return predictions
    
    def accumulate_test(self, predictions): 
        predictions = sum(predictions, [])
        accumulated_predictions = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(accumulated_predictions, predictions)
        accumulated_predictions = sum(accumulated_predictions, [])
        return accumulated_predictions

    def evaluate_test(self, predictions):
        # same identifier, merge
        heatmaps, actions = {}, {}
        print('merge predictions with same video_idx')
        for prediction in tqdm.tqdm(predictions):
            video_idx = prediction['video_idx']
            if video_idx not in heatmaps:
                heatmaps[video_idx] = prediction['heatmap']
                actions[video_idx] = prediction['action']
            else:
                heatmaps[video_idx] += prediction['heatmap']
                actions[video_idx] += prediction['action']
        
        video_idxs = heatmaps.keys()
        heatmaps = torch.cat(list(heatmaps.values()))
        gt_heatmaps = torch.cat([self.eval_gt_heatmaps[video_idx] for video_idx in video_idxs]).view_as(heatmaps)
        actions = list(actions.values())
        actions_gt = [self.eval_gt_actions[video_idx] for video_idx in video_idxs]
        count = len(heatmaps)
        
        heatmaps = heatmaps.cuda()
        gt_heatmaps = gt_heatmaps.cuda()
        from .metrics import KLD, SIM, AUC_Judd
        kld = KLD(heatmaps, gt_heatmaps).item() / count
        sim = SIM(heatmaps, gt_heatmaps).item() / count
        auc_judd = []
        for p, g in zip(heatmaps, gt_heatmaps):
            _auc_judd = AUC_Judd(p, g)
            if _auc_judd >= 0:
                auc_judd.append(_auc_judd)
        auc_judd = sum(auc_judd).item() / len(auc_judd)
        
        count, top1_acc = 0, 0
        for action, action_gt in zip(actions, actions_gt):
            for a in action.float().topk(len(action_gt)).indices.tolist():
                if a in action_gt:
                    top1_acc += 1
            count += len(action_gt)
        top1_acc /= count

        return dict(kld=kld, sim=sim, auc_judd=auc_judd, top1_acc=top1_acc)
    
    def __len__(self, ):
        return self.video_clips.num_clips()
    
if __name__ == "__main__":
    OPRA('test', -1, -1, -1, None)