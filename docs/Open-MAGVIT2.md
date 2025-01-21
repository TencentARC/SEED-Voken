<div align="center">
<h2>OPEN-MAGVIT2: An Open-source Project Toward Democratizing Auto-Regressive Visual Generation</h2>

</div>

<div align="center">

<!-- > [**OPEN-MAGVIT2: An Open-source Project Toward Democratizing Auto-Regressive Visual Generation**](https://arxiv.org/abs/2409.04410)<br> -->
> [Zhuoyan Luo*](https://robertluo1.github.io/), [Fengyuan Shi*](https://shifengyuan1999.github.io/), [Yixiao Ge](https://geyixiao.com/), [Yujiu Yang](https://sites.google.com/view/iigroup-thu/people), [Limin Wang](https://wanglimin.github.io/), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en)
> <br>ARC Lab Tencent PCG, Tsinghua University, Nanjing University<br>
</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2409.04410-b31b1b.svg)](https://arxiv.org/abs/2409.04410)&nbsp;

</div>
This is the official repository for Open-MAGVIT2, an open-source project re-implementing Google's MAGVIT-v2 tokenizer and democratizing autoregressive visual generation with a super large vocabulary (i.e., 2^18).

<p align="center">
<img src="../assets/Open-MAGVIT2-teaser.png" width=75%>
</p>

## **Highlights**
- 🚀 **Super-large Codebook**: Re-implements the advanced Lookup-Free Quantizer proposed by MAGVITv2, and achieves a super-large codebook (i.e., 2^18) with strong performance (1.17rFID).
- 💡 **Auto-Regressive Innovation**: Introduces asymmetric token factorization and the next sub-token prediction paradigm, enabling efficient generation with a super-large vocabulary and enhanced sub-token interactions.
- 🚀 **Scalability**: Validates the scalability of plain auto-regressive models across various parameter sizes (300M to 1.5B).

<p align="center">
<img src="../assets/Open-MAGVIT2-framework.png", width=75%>
</p>

This repository provides the scripts and checkpoints to replicate our results.


### 🎤 Features
* A series of image tokenizer for class-conditional image generation (8$\times$ and 16$\times$ downsampling rate with 2^18 codebook size) and text-conditional image generation (2^14 and 2^18 codebook size with 16$\times$ downsampling rate).
* A family of the autoregressive model ranging from 300M to 1.5B for class-conditional image generation.

**🤗 Open-MAGVIT2 is still under active development. Stay tuned for the update!**

---

## 🔥 Quick Start
<!-- * `Stage I Tokenizer Training`: -->

### Class Conditional Image Generation

#### Stage I: Training of Visual Tokenizer

##### 🚀 Training Scripts
- ###### $128\times 128$ Tokenizer Training
```
bash scripts/train_tokenizer/Open-MAGVIT2/run_128_L.sh MASTER_ADDR MASTER_PORT NODE_RANK
```

- ###### $256\times 256$ Tokenizer Training
```
bash scripts/train_tokenizer/Open-MAGVIT/run_256_L.sh MASTER_ADDR MASTER_PORT NODE_RANK
```
##### 🚀 Evaluation Scripts

- ###### $128\times 128$ Tokenizer Evaluation
```
bash scripts/evaluation/evaluation_128.sh
```

- ###### $256\times 256$ Tokenizer Evaluation
```
bash scripts/evaluation/evaluation_256.sh
```

##### 🍺 Performance and Models

###### Tokenizer 

| Method | Token Type | #Tokens | Train Data | Codebook Size | rFID | PSNR  | Codebook Utilization | Checkpoint |
|:------:|:----:|:-----:|:-----:|:-------------:|:----:|:----:|:---------------------:|:----:|
|Open-MAGVIT2-20240617| 2D | 16 $\times$ 16 | 256 $\times$ 256 ImageNet | 262144 | 1.53 | 21.53 | 100% | - |
|Open-MAGVIT2-20240617| 2D | 16 $\times$ 16 | 128 $\times$ 128 ImageNet | 262144 | 1.56 | 24.45 | 100% | - |
|Open-MAGVIT2| 2D | 16 $\times$ 16 | 256 $\times$ 256 ImageNet | 262144 | **1.17** | **21.90** | **100%** | [IN256_Large](https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-256-resolution/blob/main/imagenet_256_L.ckpt)|
|Open-MAGVIT2| 2D | 16 $\times$ 16 | 128 $\times$ 128 ImageNet | 262144 | **1.18** | **25.08** | **100%** |[IN128_Large](https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-128-resolution/blob/main/imagenet_128_L.ckpt)|
|Open-MAGVIT2*| 2D | 32 $\times$ 32 | 128 $\times$ 128 ImageNet | 262144 | **0.34** | **26.19** | **100%** |above|

(*) denotes that the results are from the direct inference using the model trained with $128 \times 128$ resolution without fine-tuning.
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
|Open-MAGVIT2| 343M | 16 $\times$ 16 | 3.08 | 258.26 | [AR_256_B](https://huggingface.co/TencentARC/Open-MAGVIT2-AR-B-256-resolution/blob/main/AR_256_B.ckpt)|
|Open-MAGVIT2| 804M | 16 $\times$ 16 | 2.51 | 271.70 | [AR_256_L](https://huggingface.co/TencentARC/Open-MAGVIT2-AR-L-256-resolution/blob/main/AR_256_L.ckpt)|
|Open-MAGVIT2| 1.5B | 16 $\times$ 16 | 2.33 | 271.77 | [AR_256_XL](https://huggingface.co/TencentARC/Open-MAGVIT2-AR-XL-256-resolution/blob/main/AR_256_XL.ckpt)|

### Text-conditional Image Generation

#### Stage I: Training of Visual Tokenizer

##### Data Preparation
We use LAION-COCO, CC12M, CC3M, LAION-HD, LAION-Aesthetic-umap, LAION-Aesthetic-v2 and JourneyDB for Pretraining. We recommend the data are organized in the following tar format.
```
data
└── LAION_COCO/
    ├── webdataset
        ├── 1.tar
        ├── 2.tar
        ├── 3.tar
        ├── ...
└── CC12M/
    ├── webdataset
        ├── 1.tar
        ├── 2.tar
        ├── 3.tar
        ├── ...
```
Before pretraining, the sample.json and filter_keys.json of each datasets should be prepared. Please refer to **src/Open_MAGVIT2/data/prepare_pretrain.py**

##### 🚀 Training Scripts
```
bash scripts/train_tokenizer/Open-MAGVIT2/run_256_L.sh MASTER_ADDR MASTER_PORT NODE_RANK
```

##### 🚀 Evaluation Scripts
- ###### $256\times 256$ Tokenizer Evaluation
```
bash scripts/evaluation/evaluation_256.sh
```

- ###### Original Resolution Tokenizer Evaluation
```
bash scripts/evaluation/evaluation_original.sh
```

##### 🍺 Performance comparison and Models
| Method | Quantizer Type | Training Data | Ratio | Resolution | Codebook Size | Checkpoint | rFID(COCO) | PSNR(COCO) | SSIM(COCO) | rFID(In1k) | PSNR(In1k) | SSIM(In1k) |
|:------:|:-----:|:-------:|:---:|:--:|:----------:|:--------:|:-------:|:-----:|:-----:|:-----:|:----:|:------:|
| LlamaGen | VQ | 70M | 16 | 256 $\times$ 256 | 16384 | - | 8.40 | 20.28 | 0.55 | 2.47 | 20.65 | 0.54 |
| Show-o | LFQ | 35M | 16 | 256 $\times$ 256 | 8192 | - | 9.26 | 20.90 | 0.59 | 3.50 | 21.34 | 0.59 |
| Cosmos | FSQ | - | 16 | 256 $\times$ 256 | 64000 | - | 11.97 | 19.22 | 0.48 | 4.57 | 19.93 | 0.49 |
| Open-MAGVIT2 | LFQ | 100M | 16 | 256 $\times$ 256 | 16384 | [Pretrain_256_16384](https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-16384-Pretrain/blob/main/pretrain256_16384.ckpt) |7.93 | 22.21 | 0.62 | 2.55 | 22.21 | 0.62 |
| **Open-MAGVIT2** | LFQ | 100M | 16 | 256 $\times$ 256 | 262144 | [Pretrain_256_262144](https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-262144-Pretrain/blob/main/pretrain256_262144.ckpt) | **6.76** | **22.31** | **0.65** | **1.67** | **22.70** | **0.64** |
| Cosmos | FSQ | - | 16 | Original | 64000 | - | 7.51 | 20.45 | 0.52 | 1.93 | 20.56 | 0.51 |
| Open-MAGVIT2 | LFQ | 100M | 16 | Original | 16384 | [Pretrain_256_16384](https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-16384-Pretrain/blob/main/pretrain256_16384.ckpt) | 6.65 | 21.61 | 0.57 | 1.39 | 21.74 | 0.56 |
| **Open-MAGVIT2** | LFQ | 100M | 16 | Original | 262144 | [Pretrain_256_262144](https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-262144-Pretrain/blob/main/pretrain256_262144.ckpt) | **5.10** | **22.18** | **0.60** | **0.78** | **22.24** | **0.59** |

