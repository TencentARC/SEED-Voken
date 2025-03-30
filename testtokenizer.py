import os
import torch
import importlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import argparse
from OpenImageTokenizer.Open_MAGVIT2.models import VQModel

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_vqgan_new(config, ckpt_path=None):
    model = VQModel(**config.model.init_args)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")
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

def visualize_tokens(indices, save_path, token_size=16):
    """
    Visualize the token indices as a grayscale image
    Each token is represented as a small square in the visualization
    """
    indices = indices.detach().cpu().numpy()
    # Get spatial dimensions
    h, w = indices.shape
    
    # Create a visualization image
    viz_img = np.zeros((h * token_size, w * token_size), dtype=np.uint8)
    
    # Normalize indices to the range [0, 255]
    max_idx = np.max(indices)
    min_idx = np.min(indices)
    norm_indices = ((indices - min_idx) / (max_idx - min_idx) * 255).astype(np.uint8)
    
    # Fill in the visualization
    for i in range(h):
        for j in range(w):
            viz_img[i*token_size:(i+1)*token_size, j*token_size:(j+1)*token_size] = norm_indices[i, j]
    
    # Save the visualization
    Image.fromarray(viz_img).save(save_path)
    
    # Return some token statistics
    token_stats = {
        "unique_tokens": len(np.unique(indices)),
        "total_tokens": indices.size,
        "min_token_id": min_idx,
        "max_token_id": max_idx
    }
    return token_stats

def main(args):
    # Load configuration
    config_file = args.config_file
    configs = OmegaConf.load(config_file)
    
    # Update configuration with command line arguments
    configs.data.init_args.batch_size = args.batch_size
    configs.data.init_args.test.params.config.size = args.image_size
    if args.subset:
        configs.data.init_args.test.params.config.subset = args.subset
    
    # Load the model
    model = load_vqgan_new(configs, args.ckpt_path).to(DEVICE)
    
    # Create output directories
    visualize_dir = args.save_dir
    visualize_version = args.version
    visualize_original = os.path.join(visualize_dir, visualize_version, f"original_{args.image_size}")
    visualize_rec = os.path.join(visualize_dir, visualize_version, f"rec_{args.image_size}")
    visualize_tokens = os.path.join(visualize_dir, visualize_version, f"tokens_{args.image_size}")
    
    os.makedirs(visualize_original, exist_ok=True)
    os.makedirs(visualize_rec, exist_ok=True)
    os.makedirs(visualize_tokens, exist_ok=True)
    
    # Load dataset
    dataset = instantiate_from_config(configs.data)
    dataset.prepare_data()
    dataset.setup()
    
    # Process images
    count = 0
    token_stats_all = []
    
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataset._val_dataloader())):
            if count >= args.image_num:
                break
                
            images = batch["image"].permute(0, 3, 1, 2).to(DEVICE)
            count += images.shape[0]
            
            # Process with the model
            if model.use_ema:
                with model.ema_scope():
                    quant, diff, indices, _ = model.encode(images)
                    reconstructed_images = model.decode(quant)
            else:
                quant, diff, indices, _ = model.encode(images)
                reconstructed_images = model.decode(quant)
            
            # Process each image in the batch
            for i in range(images.shape[0]):
                if count > args.image_num:
                    break
                    
                image = images[i]
                reconstructed_image = reconstructed_images[i]
                token_indices = indices[i]
                
                # Save original and reconstructed images
                image_pil = custom_to_pil(image)
                reconstructed_pil = custom_to_pil(reconstructed_image)
                
                save_idx = idx * args.batch_size + i
                image_pil.save(os.path.join(visualize_original, f"{save_idx}.png"))
                reconstructed_pil.save(os.path.join(visualize_rec, f"{save_idx}.png"))
                
                # Visualize and save token information
                token_stats = visualize_tokens(
                    token_indices, 
                    os.path.join(visualize_tokens, f"{save_idx}.png")
                )
                token_stats_all.append(token_stats)
                
                # Print token statistics
                if args.verbose:
                    print(f"Image {save_idx}:")
                    print(f"  Unique tokens: {token_stats['unique_tokens']}/{token_stats['total_tokens']}")
                    print(f"  Token ID range: {token_stats['min_token_id']} - {token_stats['max_token_id']}")
    
    # Save overall token statistics
    if token_stats_all:
        avg_unique_tokens = sum(stats["unique_tokens"] for stats in token_stats_all) / len(token_stats_all)
        print(f"\nProcessed {len(token_stats_all)} images")
        print(f"Average unique tokens per image: {avg_unique_tokens:.2f}")
    
def get_args():
    parser = argparse.ArgumentParser(description="Open-MAGVIT2 Image Tokenizer")
    parser.add_argument("--config_file", required=True, type=str, help="Path to the configuration file")
    parser.add_argument("--ckpt_path", required=True, type=str, help="Path to the model checkpoint")
    parser.add_argument("--image_size", default=256, type=int, help="Image size for processing")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size for inference")
    parser.add_argument("--image_num", default=50, type=int, help="Number of images to process")
    parser.add_argument("--subset", default=None, help="Subset of data to use")
    parser.add_argument("--version", type=str, required=True, help="Version identifier for output")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--verbose", action="store_true", help="Print detailed token statistics")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)