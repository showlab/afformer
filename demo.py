import torch, cv2, imageio
import numpy as np
from torchvision.transforms import Resize
from main import BaseModel, BaseData
from lightning.pytorch.cli import LightningCLI
from torchvision.io import read_image, ImageReadMode
from data.video_io import read_video_without_audio
from torchvision.transforms.functional import gaussian_blur

class DemoLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument('--image', default='demo/image.jpg')
        parser.add_argument('--video', default='demo/video.mp4')
        parser.add_argument('--output', default='demo/output.gif')
        parser.add_argument('--weight', default='weights/afformer_vitdet_b_v1.ckpt')

def overlay(image, heatmap, alpha=0.7):
    heatmap = 255 * heatmap / heatmap.max()
    heatmap = np.tile(heatmap[:, :, np.newaxis], (1, 1, 3))
    heatmap = heatmap.astype('uint8')
    image = image.astype('uint8')
    mask = heatmap == 0
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap[mask] = 0
    return cv2.addWeighted(heatmap, alpha, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 1- alpha, 0)

def save_to_gif(video, image, file):
    concat_frames = [np.hstack((frame, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))) for frame in video]
    imageio.mimsave(file, concat_frames, fps=30)

if __name__ == '__main__':
    cli = DemoLightningCLI(BaseModel, BaseData, parser_kwargs={'parser_mode': 'omegaconf'}, run=False)
    cli.model.load_state_dict(torch.load(cli.config.weight)['state_dict'])
    network = cli.model.network.cuda()
    network.eval()
    
    image_path, video_path, output_path = \
        cli.config.image, cli.config.video, cli.config.output

    image = read_image(image_path, ImageReadMode.RGB)
    video = read_video_without_audio(video_path, output_format='TCHW')
        
    vh, vw = 256, int(256 * video.shape[-1] / video.shape[-2])
    ih, iw = 256, 256
        
    image_to_show = Resize((ih,iw), antialias=True)(image).permute(1,2,0).numpy()
    video_to_show = Resize((vh,vw), antialias=True)(video).permute(0,2,3,1).numpy()
        
    image, video = cli.datamodule.dataset_val.resize(image).cuda(), \
            cli.datamodule.dataset_val.resize(video).cuda()
        
    # sampling according to model training fps for better performance
    frame_rate = cli.config.data.dataset_train.init_args.frame_rate
    video = video[::frame_rate]
    
    # limit the maximum video length to fit model relative embed size
    max_num_frames = cli.config.data.dataset_train.init_args.clip_length_in_frames
    interval = cli.config.data.dataset_train.init_args.frames_between_clips
    heatmap, action, i = 0, 0, 0

    # TODO: be faster
    while True:
        if i*interval >= len(video):
            break
        clip = video[i*interval:i*interval+max_num_frames]
        # image, video, num_frames_list, batch size 1
        batch = [image[None], clip, [len(clip)]]
        # batch size 1
        with torch.no_grad():
            _heatmap, _action = network(batch)
            heatmap = heatmap + _heatmap[0].reshape(-1,256,256)
            action = action + _action[0]
        i += 1

    heatmap = heatmap / i
    action = action / i
        
    # select top-1 action & top-50 points + gaussian to visualize
    heatmap = heatmap[action.argmax()]
    action = cli.datamodule.dataset_val.actions[action.argmax().item()]
    values, indices = heatmap.view(-1).topk(k=50, sorted=False)
    heatmap = torch.zeros_like(heatmap, device=heatmap.device)
    heatmap.view(-1).scatter_(0, indices, values)
    heatmap = gaussian_blur(heatmap[None], (31,31))[0]
    image_with_heatmap = overlay(image_to_show, heatmap.cpu().numpy())
    cv2.putText(image_with_heatmap, f'Top-1 action: {action}', (20, 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    save_to_gif(video_to_show, image_with_heatmap, output_path)
