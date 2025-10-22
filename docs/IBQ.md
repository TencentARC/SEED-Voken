<div align="center"> 
<h2>Scalable Image Tokenization with Index Backpropagation Quantization</h2>
</div>

<div align="center">

<!-- > [**Taming Scalable Image Tokenization with Index Backpropagation Quantization**](https://arxiv.org/abs/2412.02692)<br> -->
> [Fengyuan Shi*](https://shifengyuan1999.github.io/), [Zhuoyan Luo*](https://robertluo1.github.io/), [Yixiao Ge](https://geyixiao.com/), [Yujiu Yang](https://sites.google.com/view/iigroup-thu/people), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en), [Limin Wang](https://wanglimin.github.io/)
> <br> Nanjing University, Tsinghua University, ARC Lab Tencent PCG<br>
</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2412.02692-b31b1b.svg)](https://arxiv.org/abs/2412.02692)&nbsp;

</div>

This is the official repository for Index Backpropagation Quantization (IBQ), a novel vector quantization (VQ) method that revolutionizes the scalability and performance of visual tokenizers.
<p align="center">
<img src="../assets/IBQ-teaser.png" width=75%>
</p>

## **Highlights**
- 🚀 **Scalable Visual Tokenizers**: IBQ enables scalable training of visual tokenizers, and achieves a large-scale codebook of size (262144) and high-dimensional embeddings (256), ensuring high utilization.
- 💡 **Innovative Approach**: Unlike conventional VQ methods prone to codebook collapse due to the partial-updating, IBQ leverages a straight-through estimator on the categorical distribution, enabling the joint optimization of all codebook embeddings and the visual encoder, for consistent latent space.  
- 🏆 **Superior Performance**: Demonstrates competitive results on ImageNet:  
  - **Reconstruction**: 1.00 rFID, outperforming Open-MAGVIT2 (1.17 rFID)
  - **Autoregressive Visual Generation**: 2.05 gFID, outperforming previous vanilla autoregressive transformers.

<p align="center">
<img src="../assets/IBQ-gradient-flow.png" width=75%>
</p>

This repository provides the scripts and checkpoints to replicate our results.

## 🔥 Quick Start
<!-- * `Stage I Tokenizer Training`: -->
### Class Conditional Image Generation

#### Stage I: Training of Visual Tokenizer

##### 🚀 Training Scripts
- ###### $256\times 256$ Tokenizer Training
```
bash scripts/train_tokenizer/IBQ/run_16384.sh MASTER_ADDR MASTER_PORT NODE_RANK
```

```
bash scripts/train_tokenizer/IBQ/run_262144.sh MASTER_ADDR MASTER_PORT NODE_RANK
```

##### 🍺 Performance and Models
| Method | #Tokens |  Codebook Size  | rFID | LPIPS  | Codebook Utilization | Checkpoint |
|:------:|:-----:|:---------------:|:----:|:------:|:---------------------:|:----:|
|IBQ| 16 $\times$ 16 |      1024       | 2.24 | 0.2580 | 99% | [Tokenizer-1024](https://huggingface.co/TencentARC/IBQ-Tokenizer-1024/blob/main/imagenet256_1024.ckpt)|
|IBQ| 16 $\times$ 16 |      8192       | 1.87 | 0.2437 | 98% | [Tokenizer-8192](https://huggingface.co/TencentARC/IBQ-Tokenizer-8192/blob/main/imagenet256_8192.ckpt)|
|IBQ| 16 $\times$ 16 |      16384      | 1.37 | 0.2235 | 96% | [Tokenizer-16384](https://huggingface.co/TencentARC/IBQ-Tokenizer-16384/blob/main/imagenet256_16384.ckpt)|
|IBQ| 16 $\times$ 16 |     262144      | 1.00 | 0.2030 | 84% | [Tokenizer-262144](https://huggingface.co/TencentARC/IBQ-Tokenizer-262144/blob/main/imagenet256_262144.ckpt)|

##### 🚀 Evaluation Scripts
```
bash scripts/evaluation/evaluation_256.sh
```

#### Stage II: Training of Auto-Regressive Models

##### 🚀 Training Scripts
Please see in scripts/train_autogressive/run.sh for different model configurations.
```
bash scripts/train_autogressive/run.sh MASTER_ADDR MASTER_PORT NODE_RANK
```

##### 🚀 Sample Scripts
Please see in scripts/train_autogressive/run.sh for different sampling hyper-parameters for different scale of models.
```
bash scripts/evaluation/sample_npu.sh or scripts/evaluation/sample_gpu.sh Your_Total_Rank
```

##### 🍺 Performance and Models
| Method | Params| #Tokens | FID | IS | Checkpoint |
|:------:|:-----:|:-------:|:---:|:--:|:----------:|
|IBQ| 342M | 16 $\times$ 16 | 2.88 | 254.73 | [AR_256_B](https://huggingface.co/TencentARC/IBQ-AR-B/blob/main/AR_256_B.ckpt)|
|IBQ| 649M | 16 $\times$ 16 | 2.45 | 267.48 | [AR_256_L](https://huggingface.co/TencentARC/IBQ-AR-L/blob/main/AR_256_L.ckpt)|
|IBQ| 1.1B | 16 $\times$ 16 | 2.14 | 278.99 | [AR_256_XL](https://huggingface.co/TencentARC/IBQ-AR-XL/blob/main/AR_256_XL.ckpt)|
|IBQ| 2.1B | 16 $\times$ 16 | 2.05 | 286.73 | [AR_256_XXL](https://huggingface.co/TencentARC/IBQ-AR-XXL/blob/main/AR_256_XXL.ckpt)|


### Text-conditional Image Generation

#### Stage I: Training of Visual Tokenizer

##### Data Preparation
We use CapFusion, LAION-COCO, CC12M, CC3M, LAION-HD, LAION-Aesthetic-umap, LAION-Aesthetic-v2 and JourneyDB for Pretraining.

##### 🚀 Training Scripts
```
bash scripts/train_tokenizer/IBQ/pretrain_256.sh MASTER_ADDR MASTER_PORT NODE_RANK
```

##### 🚀 Evaluation Scripts
- ###### $256\times 256$ Tokenizer Evaluation
```
bash scripts/evaluation/evaluation_256.sh
```

##### 🍺 Performance comparison and Models
| Method | Quantizer Type | Training Data | Ratio | Resolution | Codebook Size | Checkpoint | rFID(COCO) | PSNR(COCO) | SSIM(COCO) | rFID(In1k) | PSNR(In1k) | SSIM(In1k) |
|:------:|:-----:|:-------:|:---:|:--:|:----------:|:--------:|:-------:|:-----:|:-----:|:-----:|:----:|:------:|
| LlamaGen | VQ | 70M | 16 | 256 $\times$ 256 | 16384 | - | 8.40 | 20.28 | 0.55 | 2.47 | 20.65 | 0.54 |
| Show-o | LFQ | 35M | 16 | 256 $\times$ 256 | 8192 | - | 9.26 | 20.90 | 0.59 | 3.50 | 21.34 | 0.59 |
| Cosmos | FSQ | - | 16 | 256 $\times$ 256 | 64000 | - | 11.97 | 19.22 | 0.48 | 4.57 | 19.93 | 0.49 |
| Open-MAGVIT2 | LFQ | 100M | 16 | 256 $\times$ 256 | 16384 | [Open-MAGVIT2_Pretrain_256_16384](https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-16384-Pretrain/blob/main/pretrain256_16384.ckpt) |7.93 | 22.21 | 0.62 | 2.55 | 22.21 | 0.62 |
| Open-MAGVIT2 | LFQ | 100M | 16 | 256 $\times$ 256 | 262144 | [Open-MAGVIT2_Pretrain_256_262144](https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-262144-Pretrain/blob/main/pretrain256_262144.ckpt) | 6.76 | 22.31 | 0.65 | 1.67 | 22.70 | 0.64 |
| **IBQ** | IBQ | 200M | 16 | 256 $\times$ 256 | 16384 | [IBQ_Pretrain_256_16384](https://huggingface.co/TencentARC/IBQ-Tokenizer-16384-Pretrain/blob/main/IBQ_pretrain_16384.ckpt) | 7.67 | 21.58 |0.62 | 2.06 | 22.01 | 0.61 |
| **IBQ** | IBQ | 200M | 16 | 256 $\times$ 256 | 262144 | [IBQ_Pretrain_256_262144](https://huggingface.co/TencentARC/IBQ-Tokenizer-262144-Pretrain/blob/main/IBQ_pretrain_262144.ckpt) | 6.79 | 22.28 | 0.65 | 1.53 | 22.69 | 0.64 |
<!-- | Cosmos | FSQ | - | 16 | Original | 64000 | - | 7.51 | 20.45 | 0.52 | 1.93 | 20.56 | 0.51 |
| Open-MAGVIT2 | LFQ | 100M | 16 | Original | 16384 | [Pretrain_256_16384](https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-16384-Pretrain/blob/main/pretrain256_16384.ckpt) | 6.65 | 21.61 | 0.57 | 1.39 | 21.74 | 0.56 |
| **Open-MAGVIT2** | LFQ | 100M | 16 | Original | 262144 | [Pretrain_256_262144](https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-262144-Pretrain/blob/main/pretrain256_262144.ckpt) | **5.10** | **22.18** | **0.60** | **0.78** | **22.24** | **0.59** | -->


