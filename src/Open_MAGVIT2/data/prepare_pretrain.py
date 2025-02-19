"""
Using Webdataset in Lightning 
should prepare sample.json and filter_keys.json
We use a subset of 
LAION-COCO, CC15M, LAION-Aesthetic-umap
LAION-Aesthetic-v2, JourneyDB, LAION-HD
"""
import webdataset as wds
from PIL import Image
import io
from torch.utils.data import DataLoader, default_collate
import torchvision.transforms as T
import os
import json
from omegaconf import OmegaConf
from tqdm import tqdm
import os
import tarfile
import pandas as pd
from PIL import Image
import io
import json
import multiprocessing as mp
import datetime
import warnings
warnings.simplefilter("always")

def check_image(image_data, filter=True):
    save_image = False
    try:
        with warnings.catch_warnings(record=True) as w:
            image = Image.open(io.BytesIO(image_data))
            if filter:
                w, h = image.size
                if w >= 512 and h >= 512: ## filter low resolution and aspect ratio > 2
                    horizational_aspect_ratio = w // h
                    vertical_aspect_ratio = h // w
                    if horizational_aspect_ratio > 2 or vertical_aspect_ratio > 2:
                        save_image = False
                    else:
                        save_image = True
            else:
                save_image = True
            if w:
                save_image = False
                print(f"warning: {w[0].message}")
            else:
                save_image = True
        return save_image
    except Exception as e:
        print(f"Error details: {str(e)}")
        save_image = False
    return save_image

def check_tar_file(args):
    tar_dict = dict()
    filter_keys = dict()
    bad_tar_file = []
    tar_paths, num_processes_idx, unit = args[0], args[1], args[2]
    for idx, tar_path in enumerate(tar_paths):
        cnt = 0
        temp_filter_keys = []
        with tarfile.open(os.path.join(tar_path), "r") as tar:
            try:
                members = tar.getmembers()
            except Exception as e:
                print(f"Error details: {str(e)}")
                print("skip the" + tar_path)
                bad_tar_file.append(tar_path)
                continue
            for member in members:
                if member.isfile():
                    name = member.name
                    if name.endswith(".jpg"):
                        image_data = tar.extractfile(member).read()
                        check = check_image(image_data, filter=False)
                        if check:
                            name, ext = os.path.splitext(name) #name, jpg
                            cnt +=1
                        else:
                            temp_filter_keys.append(name)
                            continue
        filter_keys[tar_path] = temp_filter_keys
        tar_dict[tar_path] = cnt
        print(f"[{datetime.datetime.now()}] complete to check in {(num_processes_idx * unit + idx)}")
    return tar_dict, filter_keys, bad_tar_file

if __name__ == "__main__":
    ### The datasets should be in the format
    ### {1..n}.tar

    TAR_DIR = "../../data/tar_dirs" ##please specify your own datasets
    filter_keys = dict()
    tar_dicts = dict()
    bad_tar_files = []

    tar_paths = [os.path.join(TAR_DIR, name) for name in os.listdir(TAR_DIR) if name.endswith("tar")]
    num_processes = int(max(mp.cpu_count(), 4) * 0.8)
    unit = len(tar_paths) // num_processes + 1
    work_list = [(tar_paths[idx*unit:(idx+1)*unit], idx, unit) for idx in range(num_processes)]
    with mp.Pool(processes=num_processes) as pool:
        result = pool.map(check_tar_file, work_list)

    for sublist in result:
        tar_dict, filter_key, bad_tar_file = sublist[0], sublist[1], sublist[2]
        tar_dicts.update(tar_dict)
        filter_keys.update(filter_key)
        bad_tar_files = bad_tar_files + bad_tar_file

    with open("../../data/samples.json", "w") as f:
        json.dump(tar_dicts, f)

    with open("../../data/filter_keys.json", "w") as f:
        json.dump(filter_keys, f)
