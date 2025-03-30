import os
import torch
import importlib
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import argparse
from OpenImageTokenizer.Open_MAGVIT2.models.lfqgan import VQModel

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def load_model(config_path, ckpt_path):
    """
    Load model from configuration and checkpoint
    Supports both YAML and JSON configurations
    """
    # Determine if config is YAML or JSON
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        config = OmegaConf.load(config_path)
        model = VQModel(**config.model.init_args)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        model = VQModel(**config_dict["model_params"])
    else:
        raise ValueError(f"Unsupported configuration format: {config_path}")
    
    # Load checkpoint
    if ckpt_path is not None:
        print(f"Loading checkpoint from {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")
        # Handle different checkpoint formats
        if "state_dict" in sd:
            sd = sd["state_dict"]
        
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
    print(f"Token indices shape before processing: {indices.shape if hasattr(indices, 'shape') else 'unknown'}")
    print(f"Token indices type: {type(indices)}")
    
    # Ensure indices is a tensor and detach if needed
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu()
    
    # Handle 1D tensors by reshaping to 2D
    if isinstance(indices, torch.Tensor) and indices.dim() == 1:
        # Try to infer a square-ish shape for the tokens
        size = int(np.sqrt(indices.shape[0]))
        indices = indices.reshape(size, -1)
        print(f"Reshaped 1D tensor to 2D: {indices.shape}")
    
    # Convert to numpy array
    if isinstance(indices, torch.Tensor):
        indices = indices.numpy()
    
    print(f"Token indices shape after processing: {indices.shape}")
    
    # Get spatial dimensions
    h, w = indices.shape
    
    # Create a visualization image
    viz_img = np.zeros((h * token_size, w * token_size), dtype=np.uint8)
    
    # Handle empty indices or constant values
    max_idx = np.max(indices)
    min_idx = np.min(indices)
    
    # If all indices are the same, use a constant color
    if max_idx == min_idx:
        viz_img.fill(128)  # Use a mid-gray for constant indices
        print("All token indices are the same value")
    else:
        # Normalize indices to the range [0, 255]
        norm_indices = ((indices - min_idx) / (max_idx - min_idx) * 255).astype(np.uint8)
        
        # Fill in the visualization
        for i in range(h):
            for j in range(w):
                viz_img[i*token_size:(i+1)*token_size, j*token_size:(j+1)*token_size] = norm_indices[i, j]
    
    # Create a colored version for better visualization
    # Map to a colormap like jet for more visual distinction between tokens
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        # Create a colormap version
        if max_idx != min_idx:
            norm_float = (indices - min_idx) / (max_idx - min_idx)
            colored = cm.viridis(norm_float)
            colored = (colored * 255).astype(np.uint8)
            
            color_img = np.zeros((h * token_size, w * token_size, 4), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    color_img[i*token_size:(i+1)*token_size, j*token_size:(j+1)*token_size] = colored[i, j]
            
            # Save both grayscale and colored versions
            Image.fromarray(viz_img).save(save_path)
            
            color_path = save_path.replace('.png', '_color.png')
            Image.fromarray(color_img[:,:,:3]).save(color_path)
            print(f"Saved colored token visualization to: {color_path}")
        else:
            # Just save the grayscale for constant values
            Image.fromarray(viz_img).save(save_path)
            
    except ImportError:
        # If matplotlib is not available, just save the grayscale version
        Image.fromarray(viz_img).save(save_path)
    
    # Return some token statistics
    token_stats = {
        "unique_tokens": len(np.unique(indices)),
        "total_tokens": indices.size,
        "min_token_id": min_idx,
        "max_token_id": max_idx,
        "token_shape": f"{h}x{w}"
    }
    
    print(f"Token statistics: {token_stats}")
    return token_stats

def main(args):
    # Load the model directly
    model = load_model(args.config_file, args.ckpt_path).to(DEVICE)
    print(f"Model loaded successfully")
    
    # Create output directories
    visualize_dir = args.save_dir
    os.makedirs(visualize_dir, exist_ok=True)
    
    visualize_original = os.path.join(visualize_dir, "original")
    visualize_rec = os.path.join(visualize_dir, "reconstructed")
    visualize_tokens = os.path.join(visualize_dir, "tokens")
    
    os.makedirs(visualize_original, exist_ok=True)
    os.makedirs(visualize_rec, exist_ok=True)
    os.makedirs(visualize_tokens, exist_ok=True)
    
    # Process single image or directory
    if args.image_path:
        # Process a single image
        if os.path.isfile(args.image_path):
            process_single_image(args.image_path, model, visualize_original, visualize_rec, visualize_tokens, args.verbose)
        # Process a directory of images
        elif os.path.isdir(args.image_path):
            image_files = [os.path.join(args.image_path, f) for f in os.listdir(args.image_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for idx, img_path in enumerate(tqdm(image_files[:args.image_num])):
                process_single_image(img_path, model, visualize_original, visualize_rec, visualize_tokens, args.verbose, idx)
        else:
            print(f"Error: {args.image_path} is not a valid file or directory")
    elif args.use_dataset:
        # Process dataset from config
        try:
            config = OmegaConf.load(args.config_file)
            config.data.init_args.batch_size = args.batch_size
            config.data.init_args.test.params.config.size = args.image_size
            if args.subset:
                config.data.init_args.test.params.config.subset = args.subset
            
            # Load dataset
            dataset = instantiate_from_config(config.data)
            dataset.prepare_data()
            dataset.setup()
            
            # Process images from dataset
            count = 0
            token_stats_all = []
            
            with torch.no_grad():
                for idx, batch in tqdm(enumerate(dataset._val_dataloader())):
                    if count >= args.image_num:
                        break
                        
                    images = batch["image"].permute(0, 3, 1, 2).to(DEVICE)
                    count += images.shape[0]
                    
                    # Process with the model
                    if hasattr(model, 'use_ema') and model.use_ema:
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
        except Exception as e:
            print(f"Error processing dataset: {e}")
    else:
        print("Error: Either --image_path or --use_dataset must be specified")

def process_single_image(image_path, model, orig_dir, rec_dir, token_dir, verbose=False, idx=0):
    """Process a single image file with the model"""
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_name = os.path.basename(image_path)
        
        # Default size if not specified in model
        size = 256
        
        # Get image size from model if available
        if hasattr(model, 'image_size'):
            size = model.image_size
        
        # Resize image
        if img.size != (size, size):
            print(f"Resizing image from {img.size} to ({size}, {size})")
            img = img.resize((size, size), Image.LANCZOS)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(np.array(img)).float()
        img_tensor = img_tensor / 127.5 - 1.0  # Normalize to [-1, 1]
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # Add batch dimension
        
        if verbose:
            print(f"Image tensor shape: {img_tensor.shape}")
        
        with torch.no_grad():
            # Print model attributes for debugging
            if verbose:
                print(f"Model type: {type(model).__name__}")
                print(f"Model has EMA: {hasattr(model, 'use_ema') and model.use_ema}")
            
            try:
                # Encode the image
                if hasattr(model, 'use_ema') and model.use_ema:
                    with model.ema_scope():
                        encode_output = model.encode(img_tensor)
                else:
                    encode_output = model.encode(img_tensor)
                
                # Handle different output formats from model.encode()
                if isinstance(encode_output, tuple) and len(encode_output) >= 3:
                    # For outputs like (quant, diff, indices, _)
                    quant = encode_output[0]
                    indices = encode_output[2]  # Get indices from the 3rd position
                elif isinstance(encode_output, dict):
                    # For dictionary outputs
                    quant = encode_output.get('quant')
                    indices = encode_output.get('indices') 
                else:
                    raise ValueError(f"Unexpected encode output format: {type(encode_output)}")
                
                # Decode the quantized representation
                if hasattr(model, 'use_ema') and model.use_ema:
                    with model.ema_scope():
                        reconstructed = model.decode(quant)
                else:
                    reconstructed = model.decode(quant)
            
            except Exception as model_error:
                print(f"Error during model processing: {model_error}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                raise
            
            # Save original image
            save_path = os.path.join(orig_dir, f"{idx}_{img_name}")
            img.save(save_path)
            print(f"Saved original image to: {save_path}")
            
            # Save reconstructed image
            try:
                reconstructed_img = custom_to_pil(reconstructed[0])
                save_path = os.path.join(rec_dir, f"{idx}_{img_name}")
                reconstructed_img.save(save_path)
                print(f"Saved reconstructed image to: {save_path}")
            except Exception as rec_error:
                print(f"Error saving reconstructed image: {rec_error}")
            
            # Visualize tokens
            try:
                token_indices = indices[0] if isinstance(indices, torch.Tensor) and indices.dim() > 1 else indices
                save_path = os.path.join(token_dir, f"{idx}_{os.path.splitext(img_name)[0]}_tokens.png")
                token_stats = visualize_tokens(token_indices, save_path)
                print(f"Saved token visualization to: {save_path}")
                
                if verbose:
                    print(f"Image {img_name}:")
                    print(f"  Unique tokens: {token_stats['unique_tokens']}/{token_stats['total_tokens']}")
                    print(f"  Token ID range: {token_stats['min_token_id']} - {token_stats['max_token_id']}")
                    
                return token_stats
            except Exception as token_error:
                print(f"Error visualizing tokens: {token_error}")
                if verbose:
                    import traceback
                    traceback.print_exc()
            
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None
    
def get_args():
    parser = argparse.ArgumentParser(description="Open-MAGVIT2 Image Tokenizer")
    parser.add_argument("--config_file", required=True, type=str, 
                        help="Path to the configuration file (YAML or JSON)")
    parser.add_argument("--ckpt_path", required=True, type=str, 
                        help="Path to the model checkpoint")
    parser.add_argument("--save_dir", type=str, default="./results", 
                        help="Directory to save results")
    
    # Single image or directory processing
    parser.add_argument("--image_path", type=str, 
                        help="Path to an image or directory of images to process")
    
    # Dataset processing options
    parser.add_argument("--use_dataset", action="store_true", 
                        help="Use dataset from config instead of individual images")
    parser.add_argument("--image_size", default=256, type=int, 
                        help="Image size for dataset processing")
    parser.add_argument("--batch_size", default=1, type=int, 
                        help="Batch size for inference")
    parser.add_argument("--image_num", default=50, type=int, 
                        help="Maximum number of images to process")
    parser.add_argument("--subset", default=None, 
                        help="Subset of data to use")
    
    parser.add_argument("--verbose", action="store_true", 
                        help="Print detailed token statistics")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)