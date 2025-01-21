"""
Modified from
https://github.com/webdataset/webdataset/issues/346
"""

import torch
import lightning as L
from omegaconf import OmegaConf
import json
import torchvision.transforms as T
import webdataset as wds
import os
from PIL import Image
import io
import numpy as np
from torch.utils.data import DataLoader,get_worker_info
import random
from multiprocessing import Value
from copy import deepcopy

from torch.distributed import get_rank, get_world_size

def detshuffle(epoch, tar_paths):
    assert isinstance(epoch, SharedEpoch)
    epoch = epoch.get_value()
    rng = random.Random()
    seed = pytorch_worker_seed(epoch)
    rng.seed(seed)
    rng.shuffle(tar_paths)
    return tar_paths

def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

class Iter_ds(torch.utils.data.IterableDataset):
    def __init__(self, configs, image_transform, n_sample, filter_data, tar_paths, epoch=0):
        self.configs = configs
        self.image_transform = image_transform
        self.n_sample = n_sample
        self.filter_data = filter_data
        self.tar_paths = tar_paths
        self.rng = random.Random() ##use to shuffle the shards
        self.epoch = epoch

    def __len__(self):
        # let's say i have 100 image totally, 2 gpus, batch_size = 4.
        # then n_step per epoch should be : `100 // (2 * 4) = 17`, and last batch doesn't fill with 4 samples.
        # in here, we directly control how many n_step by __len__. 
        # for the above example, in here should be `100 // 2`, then torch dataloader will try to divided this number by batch_size automatically ~ (since we setup batch_size in torch dataloader)  
        # return self.n_sample // get_world_size()
        return self.n_sample // get_world_size()
    
    def create_webdataset(
        self,
        tar_path,
        filter_data=None
    ):
        dataset = wds.WebDataset(tar_path, resampled=True, nodesplitter=wds.split_by_worker, cache_size=20 ** 20, handler=wds.handlers.warn_and_continue, shardshuffle=True) ##Can not set Resampled=True or it will always train on a single tar
        def filter_dataset(item):
            image_key = item["__key__"]
            url = item["__url__"]
            if filter_data is not None: filter_tar = filter_data.get(url, None)
            else: filter_tar = None
            if filter_tar is not None: ## a list
                for filter_value in filter_tar:
                    if image_key in filter_value:
                        return False
            return True

        def preprocess_dataset(item):
            output = {}
            if self.configs["enable_image"]:
                image_keys = self.configs["image_key"]
                for image_key in image_keys:
                    if image_key in item.keys(): ## control jpg, jpg.jpg, jpg.jpeg
                        image_data = item[image_key]
                        break
                image = Image.open(io.BytesIO(image_data))
                if not image.mode == "RGB":
                    image = image.convert("RGB")
                if self.image_transform:
                    image = self.image_transform(image) #PIL 
                image = np.array(image)
                image = (image/127.5 - 1.0).astype(np.float32)
                output["image_filename"] = item["__key__"]
                output["image"] = image

            return output
        
        dataset = dataset.select(filter_dataset)
        dataset = dataset.map(preprocess_dataset)
        dataset = dataset.shuffle(1000)

        return dataset

    def __iter__(self):
        process_rank = get_rank()
        world_size = get_world_size()
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        # seed = pytorch_worker_seed(epoch)
        # self.rng.seed(seed) ## for determinastic
        shuffle_urls = detshuffle(self.epoch, deepcopy(self.tar_paths)) # shuffle across shards
        dataset = self.create_webdataset(shuffle_urls, self.filter_data)
        for batch_id, sample in enumerate(dataset):
            # assign a independent batch for the gpu wrt. gpu_id (nodesplitter in here)
            if batch_id % world_size == process_rank: 
                yield sample
            # skip the batch it doesn't belong to the gpu
            else:
                continue

class LAIONCombineTrain(L.LightningDataModule):
    """
    Pretrain WebDataset 
    (LAIONCOCO + CC12M + CC3M)
    (LAION-aesthetic + LAION-aesthetic v2 + JourneyDB LAION-HD)
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)

        data_dirs = self.config["data_dir"] ## to caption
        self.tars_paths = []
        for data_dir in data_dirs:
            if "LAION-COCO" in data_dir: ##handle laion-coco dataset
                sample_urls = self.config.get("sample_coco_urls", None) ##file list txt
                if sample_urls is not None:
                    with open(sample_urls, "r") as f:
                        sampled_tars_paths = f.read().splitlines()
                        self.tars_paths.extend(sampled_tars_paths)
                else:
                    tars_path = [] # not subset
                    ### LAIONCOCO has different chapters and parts
                    for chap in os.listdir(data_dir):
                        for parts in os.listdir(os.path.join(data_dir, chap)):
                            for tar_name in os.listdir(os.path.join(data_dir, chap, parts)):
                                if tar_name.endswith(".tar"):
                                    tars_path.append(os.path.join(data_dir, chap, parts, tar_name))
                    self.tars_paths.extend(tars_path)
            
            ## handle CC and laion-aesthetic Dataset
            elif "laion-hd" in data_dir:
                sample_urls = self.config.get("sample_hd_urls", None)
                if sample_urls is not None:
                    with open(sample_urls, "r") as f:
                        sampled_tars_paths = f.read().splitlines()
                        self.tars_paths.extend(sampled_tars_paths)
                else:
                    tars_path = []
                    for part in os.listdir(data_dir):
                        for tar_path in os.listdir(os.path.join(data_dir, part)):
                            if tar_path.endswith(".tar"):
                                tars_path.append(os.path.join(data_dir, part, tar_path))
                    self.tars_paths.extend(tars_path)
            else:## other datasets
                for tar_name in os.listdir(data_dir):
                    if tar_name.endswith(".tar"):
                        self.tars_paths.append(os.path.join(data_dir, tar_name))
        
        filter_path = config["filter_path"] ## no need to filtering
        self.filter_data = dict()
        if filter_path is not None:
            for filter_pa in filter_path:
                with open(filter_pa, "r") as f:
                    filter_data = json.load(f)
                self.filter_data.update(filter_data)
        else:
            self.filter_data = None

        self.image_transform = self.get_transforms()

        sample_json_paths = config["sample_json_path"]
        self.tar_dict = dict()
        for sample_json_path in sample_json_paths:
            with open(sample_json_path, "r") as f:
                samples = json.load(f)
                self.tar_dict.update(samples)

        ## setup Epoch
        self.shared_epoch = SharedEpoch(epoch=0) ##The beginning

    def _get_sample_num(self, tar_list):
        cnt = 0
        for tar_key in tar_list: 
            cnt += self.tar_dict[tar_key]
        return cnt
    
    def get_transforms(self, keep_ratio=True):
        image_size = self.config["size"]
        transform = []
        if keep_ratio:
            transform.extend([
                T.Resize(image_size),
                T.CenterCrop((image_size, image_size))
            ])
        else:
            transform.append(T.Resize((image_size, image_size)))
        
        return T.Compose(transform)

    def create_dataset(self):
        n_sample = self._get_sample_num(self.tars_paths)
        dataset = Iter_ds(self.config, self.image_transform, n_sample, self.filter_data, self.tars_paths, epoch=self.shared_epoch)

        return dataset