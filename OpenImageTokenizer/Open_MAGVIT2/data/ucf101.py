import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import glob
import math
import torch.nn.functional as F
from omegaconf import OmegaConf
import numpy as np
import Open_MAGVIT2.data.video_transforms as video_transforms
import Open_MAGVIT2.data.volume_transforms as volume_transforms
from decord import VideoReader, cpu
import warnings
import random

def get_parent_dir(path):
    return os.path.basename(os.path.dirname(path))

class VideoDataset(Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-1.0, 1.0]
    use RandomClip and Resize and RandomCrop for Pre-processing 
    """
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, config=None):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        data_folder = self.config["data_folder"]
        self.mode = self.config["mode"]
        self.sequence_length = self.config["sequence_length"]
        self.resolution = self.config["size"]
        self.frame_sample_rate = self.config["frame_sample_rate"]
        subset = self.config.get("subset", None)
        num_eval_samples = self.config.get("num_eval_sample", 10000)
        self.fps = self.config.get("fps", None)

        folder = os.path.join(data_folder, "train") #only use training set
        if subset is not None:
            with open("../../data/ucf-101_{}.txt".format(subset), "r") as f:
                self.video_files = f.read().splitlines()
        else:
            self.video_files = sum([glob.glob(os.path.join(folder, '**', f'*.{ext}'), recursive=True)
                        for ext in self.exts], [])
            if self.mode != "train": ##validation randomly add video files
                left = num_eval_samples - len(self.video_files)
                random_samples = random.sample(self.video_files, left)
                self.video_files.extend(random_samples)

        # hacky way to compute # of classes (count # of unique parent directories)
        self.classes = list(set([get_parent_dir(f) for f in self.video_files]))
        self.classes.sort()
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}
        
        ##build transforms training and evaluation is the same
        self.transforms = video_transforms.Compose([
            video_transforms.Resize(self.resolution, interpolation="bilinear"),
            video_transforms.RandomCrop(size=(self.resolution, self.resolution)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ##adopted [0.5 rules]
        ])

    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):

        sample = self.video_files[idx]
        if self.mode == "train":
            frames, _ = self.load_video_from_path_decord(
                sample, 
                frm_sampling_strategy="rand",
                fps = self.fps if self.fps is not None else -1,
                num_frm=self.sequence_length
            )
        else:
            frames, _ = self.load_video_from_path_decord(
                sample,
                frm_sampling_strategy = "center",
                fps = self.fps if self.fps is not None else -1,
                num_frm = self.sequence_length
            )

        video = self.transforms(frames)

        class_name = get_parent_dir(sample) # C T H W
        label = self.class_to_label[class_name]
        return dict(video=video, label=label)
    
    def load_video_from_path_decord(
        self,
        video_path,
        frm_sampling_strategy,
        height=None,
        width=None,
        start_time=None,
        end_time=None,
        fps=-1,
        num_frm=None,
    ):
        specified_num_frm = num_frm
        if not height or not width:
            vr = VideoReader(rf"{video_path}")
        else:
            vr = VideoReader(video_path, width=width, height=height)
        
        default_fps = vr.get_avg_fps()
        if default_fps <= fps:
            fps = -1

        if fps != -1:
            # resample the video to the specified fps
            duration = len(vr) / default_fps
            num_frames_to_sample = int(duration * fps)
            resample_indices = np.linspace(
                0, len(vr) - 1, num_frames_to_sample
            ).astype(int)
            
            # print(default_fps, fps, resample_indices)
            sampled_frms = vr.get_batch(resample_indices).asnumpy().astype(np.uint8)
            default_fps = fps
            

        else:
            sampled_frms = vr.get_batch(np.arange(0, len(vr), 1, dtype=int)).asnumpy().astype(np.uint8)

        vlen = sampled_frms.shape[0]

        if num_frm is None:
            num_frm = vlen

        num_frm = min(num_frm, vlen)

        if start_time or end_time:
            assert (
                fps > 0
            ), "must provide video fps if specifying start and end time."
            start_idx = min(int(start_time * fps), vlen)
            end_idx = min(int(end_time * fps), vlen)

        else:
            start_idx, end_idx = 0, vlen

        if frm_sampling_strategy == "uniform":
            frame_indices = np.linspace(0, vlen - 1, num_frm).astype(int)

        elif frm_sampling_strategy == "nlvl_uniform":
            frame_indices = np.arange(
                start_idx, end_idx, vlen / num_frm
            ).astype(int)

        elif frm_sampling_strategy == "nlvl_rand":
            frame_indices = np.arange(
                start_idx, end_idx, vlen / num_frm
            ).astype(int)

            strides = [
                frame_indices[i] - frame_indices[i - 1]
                for i in range(1, len(frame_indices))
            ] + [vlen - frame_indices[-1]]
            pertube = np.array(
                [np.random.randint(0, stride) for stride in strides]
            )

            frame_indices = frame_indices + pertube

        elif frm_sampling_strategy == "rand":
            # frame_indices = sorted(random.sample(range(vlen), num_frm))
            rand_start = random.randint(0, vlen - num_frm)
            frame_indices = np.array(range(rand_start, rand_start + num_frm)).astype(int)
        
        elif frm_sampling_strategy == "center":
            center = vlen // 2
            if num_frm % 2 ==0:
                frame_indices = np.array(range(center - num_frm // 2, center + num_frm // 2)).astype(int)
            else:
                frame_indices = np.array(range(center - num_frm // 2, center + num_frm // 2 + 1)).astype(int)
        
        elif frm_sampling_strategy == "headtail":
            frame_indices_head = sorted(
                random.sample(range(vlen // 2), num_frm // 2)
            )
            frame_indices_tail = sorted(
                random.sample(range(vlen // 2, vlen), num_frm // 2)
            )
            frame_indices = frame_indices_head + frame_indices_tail

        elif frm_sampling_strategy == "all":
            frame_indices = np.arange(0, vlen).astype(int)

        else:
            raise NotImplementedError(
                "Invalid sampling strategy {} ".format(frm_sampling_strategy)
            )

        raw_sample_frms = sampled_frms[
            frame_indices
        ]

        if specified_num_frm is None:
            masks = np.ones(len(raw_sample_frms), dtype=np.uint8)

        # pad the video if the number of frames is less than specified
        elif len(raw_sample_frms) < specified_num_frm:
            prev_length = len(raw_sample_frms)
            zeros = np.zeros(
                (specified_num_frm - prev_length, height, width, 3),
                dtype=np.uint8,
            )
            raw_sample_frms = np.concatenate((raw_sample_frms, zeros), axis=0)
            masks = np.zeros(specified_num_frm, dtype=np.uint8)
            masks[:prev_length] = 1

        else:
            masks = np.ones(specified_num_frm, dtype=np.uint8)

        return raw_sample_frms, masks