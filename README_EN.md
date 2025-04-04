<h1 align="center">OpenImageTokenizer 🖼️→🔢</h1>

<div align="center">

**An elegant Python interface for SEED-Voken visual tokenizers**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-APACHE2.0-green.svg)](LICENSE)

</div>

## 📝 Description

OpenImageTokenizer is a Python library that provides a simplified and accessible interface for the powerful visual tokenizers developed by TencentARC in their [SEED-Voken](https://github.com/TencentARC/SEED-Voken) project. This package facilitates the use of advanced models such as Open-MAGVIT2 and IBQ without the need for complex configurations or specialized development environments.

Similar to how text tokenizers convert text into discrete tokens, visual tokenizers convert images into discrete representations (tokens) that can be used for various purposes, from compression to autoregressive image generation.

<p align="center">
<img src="https://raw.githubusercontent.com/TencentARC/SEED-Voken/main/assets/comparsion.png" width=90%>
<br><small><i>Comparison of different visual tokenizers (image from SEED-Voken)</i></small>
</p>

## ✨ Features

- **Simplified interface**: Intuitive API to use visual tokenizers without needing to understand their internal complexity
- **Automatic download**: Transparent management of checkpoints from Hugging Face without manual intervention
- **Built-in configurations**: No external YAML or JSON files required
- **Token visualization**: Tools to visualize and understand the generated tokens
- **Compatible with multiple models**: Support for different versions of Open-MAGVIT2 and IBQ
- **Cross-platform**: Works on CPU and GPU without special configurations

## 📊 Supported Models

OpenImageTokenizer provides access to the following advanced SEED-Voken models:

### Open-MAGVIT2

State-of-the-art visual tokenizer with superior performance (`0.39 rFID` for 8x downsampling).

- TencentARC/Open-MAGVIT2-Tokenizer-128-resolution
- TencentARC/Open-MAGVIT2-Tokenizer-256-resolution
- TencentARC/Open-MAGVIT2-Tokenizer-16384-Pretrain
- TencentARC/Open-MAGVIT2-Tokenizer-262144-Pretrain

### IBQ

Scalable visual tokenizer with high code dimension and high utilization.

- TencentARC/IBQ-Tokenizer-16384
- TencentARC/IBQ-Tokenizer-32768

## 🛠️ Installation

```bash
pip install OpenImageTokenizer
```

Or directly from the repository:

```bash
git clone https://github.com/F4k3r22/OpenImageTokenizer.git
cd OpenImageTokenizer
pip install -e .
```

## 🚀 Quick Usage

### Basic Example

```python
from OpenImageTokenizer import MAGVIT2ImageTokenizer

# Initialize tokenizer (automatic checkpoint download)
tokenizer = MAGVIT2ImageTokenizer("TencentARC/Open-MAGVIT2-Tokenizer-256-resolution")

# Tokenize an image
encoded = tokenizer.encode("path/to/image.jpg")
tokens = encoded['indices']

# Reconstruct the image from tokens
reconstructed = tokenizer.decode(encoded['quant'])

# Visualize the tokens
tokenizer.visualize_tokens(tokens, save_path="tokens_visualization.png")
```

### Complete Processing

```python
# Encode, decode, and visualize in a single step
results = tokenizer.process_image("path/to/image.jpg", "output/directory")

print(f"Original image: {results['original']}")
print(f"Reconstructed image: {results['reconstructed']}")
print(f"Token visualization: {results['tokens']}")
```

## 🔍 Applications

Visual tokenizers have multiple applications in computer vision and AI:

- **Autoregressive image generation**: Foundation for GPT-like models but for images
- **Multimodal models**: Connection point between language models and visual content
- **Image compression**: Efficient representation through discrete tokens
- **Semantic editing**: Token-level manipulation for controlled editing
- **Visual generation research**: Experimentation with different architectures

## 🧩 Main Components

- **MAGVIT2ImageTokenizer**: Main class for tokenization with Open-MAGVIT2
- **hf_utils**: Module for managing model downloads from Hugging Face
- **configs**: Built-in configurations for different models
- **visualize_tokens**: Utilities for visualizing and understanding the generated tokens

## 📑 Complete Script Example

```python
import os
from OpenImageTokenizer import MAGVIT2ImageTokenizer

# Initialize tokenizer
tokenizer = MAGVIT2ImageTokenizer("TencentARC/Open-MAGVIT2-Tokenizer-256-resolution")

# Load the model
tokenizer.load_model()

# Process image (encode, visualize, reconstruct)
image_path = "my_image.jpg"
output_dir = "results"

results = tokenizer.process_image(image_path, output_dir)

# Display token information
token_shape = results["token_shape"]
print(f"Token shape: {token_shape}")
print(f"Total tokens in the image: {token_shape[0] * token_shape[1]}")

print("Generated files:")
print(f"  Original: {results['original']}")
print(f"  Reconstructed: {results['reconstructed']}")
print(f"  Token visualization: {results['tokens']}")
```

## 📚 Citations

If you use OpenImageTokenizer in your research, please consider citing the original works:

For Open-MAGVIT2:

```bibtex
@article{luo2024open,
  title={Open-MAGVIT2: An Open-Source Project Toward Democratizing Auto-regressive Visual Generation},
  author={Luo, Zhuoyan and Shi, Fengyuan and Ge, Yixiao and Yang, Yujiu and Wang, Limin and Shan, Ying},
  journal={arXiv preprint arXiv:2409.04410},
  year={2024}
}
```

For IBQ:

```bibtex
@article{shi2024taming,
  title={Taming Scalable Visual Tokenizer for Autoregressive Image Generation},
  author={Shi, Fengyuan and Luo, Zhuoyan and Ge, Yixiao and Yang, Yujiu and Shan, Ying and Wang, Limin},
  journal={arXiv preprint arXiv:2412.02692},
  year={2024}
}
```

## 🤝 Contributions

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/new-functionality`)
3. Make your changes and commit them (`git commit -m 'Add new functionality'`)
4. Push to the branch (`git push origin feature/new-functionality`)
5. Open a Pull Request

## 📄 License

This project is licensed under the APACHE 2.0 license - see the [LICENSE](LICENSE) file for details.

## ❤️ Acknowledgements

- [TencentARC](https://github.com/TencentARC) for developing [SEED-Voken](https://github.com/TencentARC/SEED-Voken) and the Open-MAGVIT2 and IBQ tokenizers
- [Hugging Face](https://huggingface.co) for hosting the pre-trained models
- The teams behind [VQGAN](https://github.com/CompVis/taming-transformers), [MAGVIT](https://github.com/google-research/magvit), [LlamaGen](https://github.com/FoundationVision/LlamaGen), [RQVAE](https://github.com/kakaobrain/rq-vae-transformer) and [VideoGPT](https://github.com/wilson1yan/VideoGPT), [OmniTokenizer](https://github.com/FoundationVision/OmniTokenizer).
