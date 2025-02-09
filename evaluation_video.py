"""
We provide Video Tokenizer Evaluation code here.
Refer to 
https://github.com/richzhang/PerceptualSimilarity
https://github.com/FoundationVision/OmniTokenizer/blob/main/vqgan_eval.py
"""

import argparse
import os
import sys
sys.path.append(os.getcwd())
import torch
from omegaconf import OmegaConf
import importlib
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from src.Open_MAGVIT2.models.video_lfqgan import VQModel
import lpips
from metrics.pytorch_i3d import InceptionI3d
from metrics.fvd import frechet_distance
from src.Open_MAGVIT2.modules.util import flatten_t_dim, unflatten_t_dim, shift_dim, preprocess

try:
    import torch_npu
except:
    pass

if hasattr(torch, "npu"):
    DEVICE = torch.device("npu:0" if torch_npu.npu.is_available() else "cpu")
else:
    DEVICE = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")

TARGET_RESOLUTION = (224, 224)
_UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)

def load_vqgan_new(config, ckpt_path=None):
    model = VQModel(**config.model.init_args)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def get_obj_from_str(string, reload=False):
    print(string)
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "class_path" in config:
        raise KeyError("Expected key `class_path` to instantiate.")
    return get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def get_args():
    parser = argparse.ArgumentParser(description="inference parameters")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--visualize_dir", type=str, default="./videos")
    parser.add_argument("--version", type=str)
    return parser.parse_args()


def main():
    args = get_args()
    config_data = OmegaConf.load(args.config_file)
    config_data.data.init_args.batch_size = 4
    config_model = load_config(args.config_file, display=False)
    model = load_vqgan_new(config_model, ckpt_path=args.ckpt_path).to(DEVICE)
    model.eval()
    model = model.to(DEVICE)
    codebook_size = config_model.model.init_args.n_embed

    # FID score related
    i3d = InceptionI3d(400, in_channels=3)
    i3d_path = "../../pretrained/i3d/i3d_pretrained_400.pt" #specify your own I3D Path
    i3d.load_state_dict(torch.load(i3d_path, map_location="cpu"), strict=True)
    i3d.eval()
    i3d = i3d.to(DEVICE)

    dataset = instantiate_from_config(config_data.data)
    dataset.prepare_data()
    dataset.setup()
    pred_xs = []
    pred_recs = []

    # LPIPS score related
    loss_fn_alex = lpips.LPIPS(net='alex').to(DEVICE)  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(DEVICE)   # closer to "traditional" perceptual loss, when used for optimization
    lpips_alex = 0.0
    lpips_vgg = 0.0


    num_videos = 0
    iteration = 0
    
    #usage
    usage = {}
    for i in range(codebook_size):
        usage[i] = 0

    with torch.no_grad():
        for batch in tqdm(dataset._val_dataloader()):
            iteration += 1
            videos = model.get_input(batch, model.image_key).to(DEVICE)
            b, c, t, h, w = videos.shape
            num_videos += videos.shape[0]

            # reconstructed_videos, _, _ = model(videos)
            if model.use_ema:
                with model.ema_scope():
                    quant, diff, indices, _ = model.encode(videos)
                    reconstructed_videos = model.decode(quant)
            else:
                quant, diff, indices, _ = model.encode(videos)
                reconstructed_videos = model.decode(quant)

            reconstructed_videos = reconstructed_videos.clamp(-1, 1)

            for index in indices:
                usage[index.item()] += 1

            # calculate lpips
            lpips_alex += loss_fn_alex(flatten_t_dim(videos), flatten_t_dim(reconstructed_videos)).sum()
            lpips_vgg += loss_fn_vgg(flatten_t_dim(videos), flatten_t_dim(reconstructed_videos)).sum()

            videos = (videos + 1) / 2
            reconstructed_videos = (reconstructed_videos + 1) / 2
            videos = videos * 255.0
            reconstructed_videos = reconstructed_videos * 255.0

            videos = shift_dim(videos.cpu(), 1, -1).byte().data.numpy()
            reconstructed_videos = shift_dim(reconstructed_videos.cpu(), 1, -1).byte().data.numpy()
            videos = preprocess(videos, TARGET_RESOLUTION).to(DEVICE)
            reconstructed_videos = preprocess(reconstructed_videos, TARGET_RESOLUTION).to(DEVICE)

            pred_x = i3d(videos).cpu()
            pred_rec = i3d(reconstructed_videos).cpu()

            pred_xs.append(pred_x)
            pred_recs.append(pred_rec)

    pred_xs = torch.cat(pred_xs, dim=0)
    pred_recs = torch.cat(pred_recs, dim=0)

    fid_value = frechet_distance(pred_xs, pred_recs)
    lpips_alex_value = lpips_alex / num_videos / t
    lpips_vgg_value = lpips_vgg / num_videos / t
    
    num_count = sum([1 for key, value in usage.items() if value > 0])
    utilization = num_count / codebook_size

    print("FVD: ", fid_value.item())
    print("LPIPS_ALEX: ", lpips_alex_value.item())
    print("LPIPS_VGG: ", lpips_vgg_value.item())
    print("utilization", utilization)

if __name__ == "__main__":
    main()