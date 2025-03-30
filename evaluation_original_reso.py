"""
We provide Tokenizer Evaluation code here.
Following Cosmos, we use original resolution to evaluate
Refer to 
https://github.com/richzhang/PerceptualSimilarity
https://github.com/mseitzer/pytorch-fid
https://github.com/NVIDIA/Cosmos-Tokenizer/blob/main/cosmos_tokenizer/image_lib.py
"""

import os
import sys
sys.path.append(os.getcwd())
import torch
try:
    import torch_npu
except: 
    pass

from omegaconf import OmegaConf
import importlib
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import linalg

from OpenImageTokenizer.Open_MAGVIT2.models.lfqgan_pretrain import VQModel
from OpenImageTokenizer.IBQ.models.ibqgan import IBQ
from metrics.inception import InceptionV3
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
import argparse

if hasattr(torch, "npu"):
    DEVICE = torch.device("npu:0" if torch_npu.npu.is_available() else "cpu")
else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## for different model configuration
MODEL_TYPE = {
    "Open-MAGVIT2": VQModel,
    "IBQ": IBQ
}

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan_new(config, model_type, ckpt_path=None, is_gumbel=False):
    model = MODEL_TYPE[model_type](**config.model.init_args) 
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

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

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def pad_images(batch, spatial_align = 16):
    """Pads a batch of images to be divisible by `spatial_align`.

    Args:
        batch: The batch of images to pad, layout BxHxWx3, in any range.
        align: The alignment to pad to.
    Returns:
        The padded batch and the crop region.
    """
    height, width = batch.shape[2:4]
    align = spatial_align
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    crop_region = [
        height_to_pad >> 1,
        width_to_pad >> 1,
        height + (height_to_pad >> 1),
        width + (width_to_pad >> 1),
    ]
    batch = torch.nn.functional.pad(
        batch,
        (width_to_pad >> 1,  width_to_pad - (width_to_pad >> 1), height_to_pad >> 1, height_to_pad - (height_to_pad >> 1), 0, 0, 0, 0),
        "constant", 0
    )
    return batch, crop_region

def unpad_images(batch, crop_region):
    """Unpads image with `crop_region`.

    Args:
        batch: A batch of numpy images, layout BxHxWxC.
        crop_region: [y1,x1,y2,x2] top, left, bot, right crop indices.

    Returns:
        np.ndarray: Cropped numpy image, layout BxHxWxC.
    """
    assert len(crop_region) == 4, "crop_region should be len of 4."
    y1, x1, y2, x2 = crop_region
    return batch[:, :, y1:y2, x1:x2]


def get_args():
    parser = argparse.ArgumentParser(description="inference parameters")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--original_reso", action="store_true")
    parser.add_argument("--model", choices=["Open-MAGVIT2", "IBQ"])

    return parser.parse_args()

def main(args):
    config_data = OmegaConf.load(args.config_file)
    config_data.data.init_args.batch_size = args.batch_size
    config_data.data.init_args.validation.params.config.original_reso = args.original_reso

    config_model = load_config(args.config_file, display=False)
    model = load_vqgan_new(config_model, model_type=args.model, ckpt_path=args.ckpt_path).to(DEVICE) #please specify your own path here
    codebook_size = config_model.model.init_args.n_embed
    
    #usage
    usage = {}
    for i in range(codebook_size):
        usage[i] = 0

    # FID score related
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx]).to(DEVICE)
    inception_model.eval()

    dataset = instantiate_from_config(config_data.data)
    dataset.prepare_data()
    dataset.setup()
    pred_xs = []
    pred_recs = []

    # SSIM score related
    ssim_value = 0.0

    # PSNR score related
    psnr_value = 0.0

    num_images = 0
    num_iter = 0

    with torch.no_grad():
        for batch in tqdm(dataset._val_dataloader()):
            images = batch["image"].permute(0, 3, 1, 2).to(DEVICE) # (B, 3, H, W)
            num_images += images.shape[0]

            ###### algin with cosmos original resolution
            images, crop_region = pad_images(images)

            if model.use_ema:
                with model.ema_scope():
                    if args.model == "Open-MAGVIT2":
                        quant, diff, indices, _ = model.encode(images)
                    elif args.model == "IBQ":
                        quant, qloss, (_, _, indices) = model.encode(images)
                    reconstructed_images = model.decode(quant)
            else:
                if args.model == "Open-MAGVIT2":
                    quant, diff, indices, _ = model.encode(images)
                elif args.model == "IBQ":
                    quant, qloss, (_, _, indices) = model.encode(images)
                reconstructed_images = model.decode(quant)

            reconstructed_images = reconstructed_images.clamp(-1, 1)

            ###### algin with cosmos
            reconstructed_images = unpad_images(reconstructed_images, crop_region)
            images = unpad_images(images, crop_region)
            
            ### usage
            for index in indices:
                usage[index.item()] += 1
            
            images = (images + 1) / 2
            reconstructed_images = (reconstructed_images + 1) / 2

            # calculate fid
            pred_x = inception_model(images)[0]
            pred_x = pred_x.squeeze(3).squeeze(2).cpu().numpy()
            pred_rec = inception_model(reconstructed_images)[0]
            pred_rec = pred_rec.squeeze(3).squeeze(2).cpu().numpy()

            pred_xs.append(pred_x)
            pred_recs.append(pred_rec)

            #calculate PSNR and SSIM
            rgb_restored = (reconstructed_images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rgb_gt = (images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rgb_restored = rgb_restored.astype(np.float32) / 255.
            rgb_gt = rgb_gt.astype(np.float32) / 255.
            ssim_temp = 0
            psnr_temp = 0
            B, _, _, _ = rgb_restored.shape
            for i in range(B):
                rgb_restored_s, rgb_gt_s = rgb_restored[i], rgb_gt[i]
                ssim_temp += ssim_loss(rgb_restored_s, rgb_gt_s, data_range=1.0, channel_axis=-1)
                psnr_temp += psnr_loss(rgb_gt_s, rgb_restored_s)
            ssim_value += ssim_temp / B
            psnr_value += psnr_temp / B
            num_iter += 1

    pred_xs = np.concatenate(pred_xs, axis=0)
    pred_recs = np.concatenate(pred_recs, axis=0)

    mu_x = np.mean(pred_xs, axis=0)
    sigma_x = np.cov(pred_xs, rowvar=False)
    mu_rec = np.mean(pred_recs, axis=0)
    sigma_rec = np.cov(pred_recs, rowvar=False)


    fid_value = calculate_frechet_distance(mu_x, sigma_x, mu_rec, sigma_rec)
    ssim_value = ssim_value / num_iter
    psnr_value = psnr_value / num_iter

    num_count = sum([1 for key, value in usage.items() if value > 0])
    utilization = num_count / codebook_size

    print("FID: ", fid_value)
    print("SSIM: ", ssim_value)
    print("PSNR: ", psnr_value)
    print("utilization", utilization)
  
if __name__ == "__main__":
    args = get_args()
    main(args)