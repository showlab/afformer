import torch, sys
sys.path.append('/Users/chenjoya/projects/afformer')
from afformer import FinegrainedDecoder

max_q_thw = (1,64,64)
max_kv_thw = (32,32,32)

decoder = FinegrainedDecoder(256, 7, True, 0.0, max_q_thw, max_kv_thw).cuda()

images = [torch.rand(2, 256, 16, 16).cuda(), torch.rand(2, 256, 32, 32).cuda(), torch.rand(2, 256, 64, 64).cuda()]
videos = torch.rand(2*32, 256, 32, 32).cuda()
num_frames_list = [32,32]

h, a = decoder(images, videos, num_frames_list)

print(torch.cuda.max_memory_allocated()/1024**3)