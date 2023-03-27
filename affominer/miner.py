import os, cv2, torch, json
from torchvision.ops import box_iou
from torch.utils.data import Dataset, DataLoader

from detectron2.utils.visualizer import Visualizer

from hsrcnn import HandStateRCNN

from pytorch_lightning import LightningDataModule, LightningModule, Trainer

class FrameQueue:
    def __init__(self, maxsize):
        self.queue = []
        self.maxsize = maxsize

    def __call__(self, x):
        self.queue.append(x)
        self.queue = self.queue[-self.maxsize:]
    
    def __len__(self):
        return len(self.queue)

class AffoMiner(LightningModule):
    def __init__(self, min_cont_affo_frames=2,
        max_side_frames=31, max_num_hands=2, 
        hand_state_nms_thresh=0.5, 
        contact_state_threshold=0.99,
        fps=5):
        super().__init__()
        self.hand_state_detector = HandStateRCNN(
            box_detections_per_img=max_num_hands)
        self.min_cont_affo_frames = min_cont_affo_frames
        self.max_side_frames = max_side_frames
        self.hand_state_nms_thresh = hand_state_nms_thresh
        self.contact_state_threshold = contact_state_threshold
        self.fps = fps
        
    def hand_state_nms(self, boxes, states, scores):
        # if has P and N overlap, then maintain the highest score one
        if len(boxes) == 2 and states.sum() == 1:
            iou = box_iou(boxes[0,None],boxes[1,None])[0]
            # scores have been ranked
            if iou > self.hand_state_nms_thresh:
                boxes, states, scores = boxes[0,None], states[0,None], scores[0,None]
        return boxes, states, scores

    def judge_contact(self, peri_frame):
        # 1. this frame should have high contact score of states == 1
        peri_scores = peri_frame['hand_scores']
        peri_states = peri_frame['hand_states']
        contacting = (peri_scores > self.contact_state_threshold) & (peri_states == 1)
        if contacting.any():
            return True
        return False

    def visualize_clip(self, clip, save_dir):
        os.makedirs(save_dir, exist_ok=False)
        for i, frame in enumerate(clip):
            v = Visualizer(frame['image'])
            hand_bboxes = frame['hand_bboxes']
            if 'hand_contacts' not in frame:
                labels = [''] * len(hand_bboxes)
            else:
                labels = ['Contact' if c else '' for c in frame['hand_contacts']]
            v = v.overlay_instances(
                boxes = hand_bboxes.cpu(),
                labels = labels
            ).get_image()
            cv2.imwrite(f'{save_dir}/{i}.jpg', v)

    def save_clip(self, clip, save_dir):
        os.makedirs(save_dir, exist_ok=False)
        hands = []
        for i, frame in enumerate(clip):
            cv2.imwrite(f'{save_dir}/{i}.jpg', frame['image'])
            hands.append(dict(
                hand_bboxes=frame['hand_bboxes'], 
                hand_states=frame['hand_states'],
                hand_scores=frame['hand_scores']
            ))
        torch.save(hands, f'{save_dir}/hands.pth')
    
    def test_step_per_video(self, video, save_dir):
        if os.path.isfile(video):
            cap = cv2.VideoCapture(video)
            interval = round(cap.get(5) / self.fps)
            is_video = True
        else:
            assert os.path.isdir(video) # epic
            interval = round(60 / self.fps) # epic is 60 fps
            is_video = False
        
        queue = FrameQueue(maxsize=self.max_side_frames)
        frame_idx = -1
        clip, side_frames, clip_idx = [], 0, 0
        gpu_flag = f'gpu{self.device.index}'

        while True:
            frame_idx += 1
            if is_video:
                ret, image = cap.read()
                if not ret:
                    break
            else:
                image = f'{video}/frame_{frame_idx+1:010}.jpg' # start from 1
                if os.path.exists(image):
                    image = cv2.imread(image)
                else:
                    break
            
            if frame_idx % interval != 0:
                continue
            
            hand_bboxes, hand_states, hand_scores = self.hand_state_detector(image)
            hand_bboxes, hand_states, hand_scores = self.hand_state_nms(hand_bboxes, hand_states, hand_scores)
            
            # judge contacting or not
            frame = dict(image=image, hand_bboxes=hand_bboxes, hand_states=hand_states, hand_scores=hand_scores)
            contacting = self.judge_contact(frame)

            # if contacting, and no clip now
            if contacting and not len(clip): 
                clip = [f for f in queue.queue] + [frame]
            elif len(clip): 
                # if contacting & has clip already, we aim to select max_side_frames without contacting
                side_frames = 0 if contacting else side_frames + 1
                clip.append(frame)
                if side_frames == self.max_side_frames:
                    clip_save_dir = os.path.join(save_dir, str(clip_idx))
                    print(f'{gpu_flag}: clip of {len(clip)} frames are generated! save them to {clip_save_dir}')
                    # self.visualize_clip(clip, clip_save_dir)
                    if not os.path.exists(clip_save_dir):
                        self.save_clip(clip, clip_save_dir)
                    clip, side_frames, clip_idx = [], 0, clip_idx+1
            queue(frame)
        
        # if has clip, then it illustrates that at the end there are still affordance. directly save
        if len(clip):
            clip_save_dir = os.path.join(save_dir, str(clip_idx))
            print(f'{gpu_flag}: clip of {len(clip)} frames are generated! save them to {clip_save_dir}')
            if not os.path.exists(clip_save_dir):
                self.save_clip(clip, clip_save_dir)
            clip, side_frames, clip_idx = [], 0, clip_idx+1

        print(f'{gpu_flag}: {save_dir} have just done! {clip_idx} clips are produced.')

    def test_step(self, batch, index):
        videos, save_dirs = batch[0]
        for video, save_dir in zip(videos, save_dirs):
            self.test_step_per_video(video, save_dir)

def get_opra_videos(root, split):
    annos = os.path.join(root, f'annotations/{split}.txt')
    with open(annos) as f:
        lines = f.readlines()
    videos = []
    for line in lines:
        channel, playlist, filename = line.split(' ')[:3]
        video = os.path.join(root, 'raw_videos', channel, playlist, filename + '.mp4')
        if os.path.exists(video) and video not in videos:
            videos.append(video)
    return videos

def to_affominer_dirs(video_paths, save_dir):
    affominer_dirs = []
    for v in video_paths:
        channel, playlist, filename = v.split('/')[-3:]
        affominer_dirs.append(os.path.join(save_dir, f'{channel}_{playlist}_{filename}'))
    return affominer_dirs

class OPRAVideoPath(Dataset):
    def __init__(self, root, split, save_dir, num_gpus):
        super().__init__()
        videos = get_opra_videos(root, split)
        save_dirs = to_affominer_dirs(videos, save_dir)
        self.videos, self.save_dirs = [], []
        for video, save_dir in zip(videos, save_dirs):
            # if not os.path.exists(save_dir):
                self.videos.append(video)
                self.save_dirs.append(save_dir)
        
        print(f'no process video data size: {len(self.videos)}')

        # split according to num_gpus
        n = len(self.videos) // num_gpus + 1
        self.videos = [self.videos[n*i:n*(i+1)] for i in range(num_gpus)]
        self.save_dirs = [self.save_dirs[n*i:n*(i+1)] for i in range(num_gpus)]
        self.num_gpus = num_gpus
    
    def __getitem__(self, index):
        return self.videos[index], self.save_dirs[index]
    
    def __len__(self):
        return self.num_gpus

def get_assistq_videos(root, split, save_dir):
    annos = json.load(open(f'{root}/assistq_{split}/{split}.json')) 
    videos = annos.keys()
    input_paths = [f'{root}/assistq_{split}/{split}/{video}/video.mp4' for video in videos]
    output_dirs = [f'{save_dir}/{video}' for video in videos]
    return input_paths, output_dirs

class AssistQVideoPath(Dataset):
    def __init__(self, root, split, save_dir, num_gpus):
        super().__init__()
        videos, save_dirs = get_assistq_videos(root, split, save_dir)
        self.videos, self.save_dirs = [], []
        for video, save_dir in zip(videos, save_dirs):
                self.videos.append(video)
                self.save_dirs.append(save_dir)
        print(f'total video data size: {len(self.videos)}')

        # split according to num_gpus
        n = len(self.videos) // num_gpus + 1
        self.videos = [self.videos[n*i:n*(i+1)] for i in range(num_gpus)]
        self.save_dirs = [self.save_dirs[n*i:n*(i+1)] for i in range(num_gpus)]
        self.num_gpus = num_gpus
    
    def __getitem__(self, index):
        return self.videos[index], self.save_dirs[index]
    
    def __len__(self):
        return self.num_gpus

def get_epic_videos(root, split, save_dir):
    train_clips = json.load(open(f'{root}/annotations.json'))[f'{split}_clips']
    videos = sorted(list(set([c['v_id'] for c in train_clips]))) # video folder
    input_paths = [f'{root}/frames/{video}' for video in videos]
    output_dirs = [f'{save_dir}/{video}' for video in videos]
    return input_paths, output_dirs

class EPICVideoPath(Dataset):
    def __init__(self, root, split, save_dir, num_gpus):
        super().__init__()
        videos, save_dirs = get_epic_videos(root, split, save_dir)
        self.videos, self.save_dirs = videos, save_dirs
        print(f'total video data size: {len(self.videos)}')

        # split according to num_gpus
        n = len(self.videos) // num_gpus + 1
        self.videos = [self.videos[n*i:n*(i+1)] for i in range(num_gpus)]
        self.save_dirs = [self.save_dirs[n*i:n*(i+1)] for i in range(num_gpus)]
        self.num_gpus = num_gpus
    
    def __getitem__(self, index):
        return self.videos[index], self.save_dirs[index]
    
    def __len__(self):
        return self.num_gpus

class VideoPath(LightningDataModule):
    def __init__(self, root, split, save_dir, num_gpus, drop_last):
        super().__init__()
        if 'opra' in root:
            self.dataset = OPRAVideoPath(root, split, save_dir, num_gpus)
        if 'assistq' in root:
            self.dataset = AssistQVideoPath(root, split, save_dir, num_gpus)
        if 'epic' in root:
            self.dataset = EPICVideoPath(root, split, save_dir, num_gpus)
        else:
            assert False
        self.drop_last = drop_last

    def test_dataloader(self, ):
        return DataLoader(self.dataset, batch_size=1, 
            shuffle=False, num_workers=1, drop_last=self.drop_last, collate_fn=lambda x:x)

if __name__ == '__main__':
    affominer = AffoMiner(fps=5, max_side_frames=31)
    num_gpus = 4
    video_path = VideoPath('../datasets/epic', 'train', 'outputs/epic_train', num_gpus, drop_last=False)
    trainer = Trainer(
        devices=num_gpus,
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_false',
        num_sanity_val_steps=0
    )
    trainer.test(model=affominer, datamodule=video_path)
