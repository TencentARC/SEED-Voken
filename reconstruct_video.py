"""
Video Reconstruction code
Refer to
https://github.com/NVIDIA/Cosmos-Tokenizer
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
import mediapy as media
import cv2 
import imageio
import random
from src.Open_MAGVIT2.models.video_lfqgan import VQModel
import src.Open_MAGVIT2.data.video_transforms as video_transforms
import src.Open_MAGVIT2.data.volume_transforms as volume_transforms
from decord import VideoReader, cpu
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

def write_video(filepath: str, video: np.ndarray, fps: int = 24) -> None:
    """Writes a video to a filepath."""
    return media.write_video(filepath, video, fps=fps)

def save_image_frame(video_path, save_dir):
    cap = cv2.VideoCapture(video_path)

    # save frame interval
    interval = 0.5 
    fps = cap.get(cv2.CAP_PROP_FPS) # get FPS
    frame_interval = int(fps * interval)

    frame_count = 0
    saved_count = 0
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image_path = os.path.join(save_dir, f"frame_{saved_count * interval}.png")  # save PNG
            imageio.imsave(frame_image_path, frame_rgb)
            saved_count += 1

        frame_count += 1

    cap.release()

def tensor2numpy(input_tensor: torch.Tensor, range_min: int = -1) -> np.ndarray:
    """
    Inputs: [C T H W]
    """
    """Converts tensor in [-1,1] to image(dtype=np.uint8) in range [0..255].

    Args:
        input_tensor: Input image tensor of Bx3xHxW layout, range [-1..1].
    Returns:
        A numpy image of layout BxHxWx3, range [0..255], uint8 dtype.
    """
    if range_min == -1:
        input_tensor = (input_tensor.float() + 1.0) / 2.0
    ndim = input_tensor.ndim
    output_image = input_tensor.clamp(0, 1).cpu().numpy()
    output_image = output_image.transpose((1,) + tuple(range(2, ndim)) + (0,))
    return (output_image * _UINT8_MAX_F + 0.5).astype(np.uint8)

def get_args():
    parser = argparse.ArgumentParser(description="inference parameters")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize_dir", type=str, default="./videos")
    parser.add_argument("--version", type=str)
    parser.add_argument("--video_num_count", type=int, default=20)
    parser.add_argument("--video_dir", type=str, default="../../data/UCF-101/test")
    return parser.parse_args()


def main():
    args = get_args()
    config_data = OmegaConf.load(args.config_file)
    config_data.data.init_args.batch_size = 4
    config_model = load_config(args.config_file, display=False)
    model = load_vqgan_new(config_model, ckpt_path=args.ckpt_path).to(DEVICE)
    model.eval()
    model = model.to(DEVICE)
    resolution = 128

    transforms = video_transforms.Compose([
            video_transforms.Resize(resolution, interpolation="bilinear"),
            video_transforms.RandomCrop(size=(resolution, resolution)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ##adopted [0.5 rules]
        ])

    iteration = 0
    count = 0
    ### prepare video data
    video_path_dirs = [dirs for dirs in os.listdir(args.video_dir)]
    video_paths = []
    each_dir_sample = 10
    for video_path_dir in video_path_dirs:
        paths = [os.path.join(args.video_dir, video_path_dir, path) for path in os.listdir(os.path.join(args.video_dir, video_path_dir))]
        sample = random.sample(paths, each_dir_sample)
        video_paths.extend(sample)
    
    with torch.no_grad():
        for video_path in video_paths:
            vr = VideoReader(rf"{video_path}")
            sampled_frms = vr.get_batch(np.arange(0, len(vr), 1, dtype=int)).asnumpy().astype(np.uint8)
            vlen = sampled_frms.shape[0]
            videos = transforms(sampled_frms)
            videos = videos.unsqueeze(0).to(DEVICE)
            iteration += 1
            b, c, t, h, w = videos.shape
            temporal_window = 17
            output_video_list = []
            for idx in tqdm(range(0, (t - 1) // temporal_window + 1)):
                start, end = idx * temporal_window, (idx + 1) * temporal_window
                input_video = videos[:, :, start:end, ...]
                if model.use_ema:
                    with model.ema_scope():
                        quant, diff, indices, _ = model.encode(input_video)
                        reconstructed_video = model.decode(quant)
                else:
                    quant, diff, indices, _ = model.encode(input_video)
                    reconstructed_video = model.decode(quant)
    
                reconstructed_video = reconstructed_video.clamp(-1, 1)
                output_video_list.append(reconstructed_video)
            
            reconstructed_videos = torch.concat(output_video_list, dim=2)
            ### visualize the videos
            visualize_dir = os.path.join(args.visualize_dir, args.version)
            if not os.path.exists(visualize_dir):
                os.makedirs(visualize_dir, exist_ok=True)
            b = videos.shape[0]
            for i in range(b):
                video = videos[i]
                reconstruct_video = reconstructed_videos[i]
                np_original_video = tensor2numpy(video)
                np_reconstruct_video = tensor2numpy(reconstruct_video)
                save_original_dir = os.path.join(visualize_dir, str(count))
                save_reconstruct_dir = os.path.join(visualize_dir, str(count))
                if not os.path.exists(save_original_dir):
                    os.makedirs(save_original_dir, exist_ok=True)
                if not os.path.exists(save_reconstruct_dir):
                    os.makedirs(save_reconstruct_dir, exist_ok=True)
                save_original_file_path = os.path.join(save_original_dir, f"original_{count}.mp4")
                save_reconstruct_file_path = os.path.join(save_reconstruct_dir, f"reconstructed_{count}.mp4")
                write_video(save_original_file_path, np_original_video)
                write_video(save_reconstruct_file_path, np_reconstruct_video)
                ### save the frame
                save_image_frame(save_original_file_path, os.path.join(visualize_dir, str(count), "original_frame"))
                save_image_frame(save_reconstruct_file_path, os.path.join(visualize_dir, str(count), "reconstruct_frame"))

if __name__ == "__main__":
    main()