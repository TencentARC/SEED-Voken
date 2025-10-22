<div align="center">
<h1>🚀 SEED-Voken: A Series of Powerful Visual Tokenizers</h1>

</div>

The project aims to provide advanced visual tokenizers for autoregressive visual generation and currently supports the following methods: <br><br>

><a href="https://arxiv.org/abs/2409.04410">Open-MAGVIT2: An Open-source Project Toward Democratizing Auto-Regressive Visual Generation</a><br>
>[Zhuoyan Luo*](https://robertluo1.github.io/), [Fengyuan Shi*](https://shifengyuan1999.github.io/), [Yixiao Ge](https://geyixiao.com/), [Yujiu Yang](https://sites.google.com/view/iigroup-thu/people), [Limin Wang](https://wanglimin.github.io/), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en)<br>
>ARC Lab Tencent PCG, Tsinghua University, Nanjing University<br>
<a href="./docs/Open-MAGVIT2.md">📚Open-MAGVIT2.md</a>
> ```
> @article{luo2024open,
>   title={Open-MAGVIT2: An Open-Source Project Toward Democratizing Auto-regressive Visual Generation},
>   author={Luo, Zhuoyan and Shi, Fengyuan and Ge, Yixiao and Yang, Yujiu and Wang, Limin and Shan, Ying},
>   journal={arXiv preprint arXiv:2409.04410},
>   year={2024}
> }
> ```

> <a href="https://arxiv.org/abs/2412.02692">IBQ: Scalable Image Tokenization with Index Backpropagation Quantization</a><br>
> [Fengyuan Shi*](https://shifengyuan1999.github.io/), [Zhuoyan Luo*](https://robertluo1.github.io/), [Yixiao Ge](https://geyixiao.com/), [Yujiu Yang](https://sites.google.com/view/iigroup-thu/people), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en), [Limin Wang](https://wanglimin.github.io/)<br>
> Nanjing University, Tsinghua University, ARC Lab Tencent PCG<br>
> <a href="./docs/IBQ.md">📚IBQ.md</a>
> ```
> @InProceedings{Shi_2025_ICCV,
>   author={Shi, Fengyuan and Luo, Zhuoyan and Ge, Yixiao and Yang, Yujiu and Shan, Ying and Wang, Limin},
>   title={Scalable Image Tokenization with Index Backpropagation Quantization},
>   booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
>   month={October},
>   year={2025},
>   pages={16037-16046}
}

> ```

<p align="center">
<img src="./assets/comparsion.png" width=90%>
</p>

## 📰 News
* **[2025.06.26]**:fire::fire::fire: **IBQ is accepted by ICCV 2025.**
* **[2025.02.14]** The pretrained version of **IBQ** visual tokenizers, which achieves SOTA performance with high code dimension is released.
* **[2025.02.09]** We release Open-MAGVIT2 Video tokenizers, which achieves SOTA performance compared to OmniTokenizer, LARP and SweetTokenizer. 
* **[2025.01.21]** Open-MAGVIT2 tokenizers (codebook size of 16384 and 262144) for text-conditional image generation are now released! They are pretrained with large-scale image-text datasets, achieving SOTA performance compared to LlamaGen, Show-o, and Cosmos.
* **[2024.11.26]** We are excited to release **IBQ**, a series of scalable visual tokenizers, which achieve a large-scale codebook (2^18) with high dimension (256) and high utilization.
* **[2024.09.09]** We release an improved version of Open-MAGVIT2 tokenizer and a family of auto-regressive models ranging from 300M to 1.5B.
* **[2024.06.17]** We release the training code of the **Open-MAGVIT2** tokenizer and checkpoints for different resolutions, **achieving state-of-the-art performance (`0.39 rFID` for 8x downsampling)** compared to VQGAN, MaskGIT, and recent TiTok, LlamaGen, and OmniTokenizer.

## 📖 Implementations

**Our codebase supports both NPU and GPU for training and inference. All experiments were conducted using the Ascend 910B for training, and we validated our models on the V100. The observed performance between the two platforms is nearly identical.**

### 🛠️ Installation
#### GPU
- **Env**: We have tested on `Python 3.8.8` and `CUDA 11.8` (other versions may also be fine).
- **Dependencies**: `pip install -r requirements.txt`

#### NPU
##### Image Version
- **Env**: `Python 3.9.16` and [`CANN 8.0.T13`](https://www.hiascend.com/en/software/cann)
- **Main Dependencies**: `torch=2.1.0+cpu` + `torch-npu=2.1.0.post3-20240523` + [`Lightning`](https://github.com/hipudding/pytorch-lightning/tree/npu_support)

##### Video Version
- **Env** `Python 3.9.16` and [`CANN 8.0.T62`](https://www.hiascend.com/en/software/cann)
- **Main Dependencies**: `torch=2.1.0+cpu` + `torch-npu=2.1.0.post10.dev20241128` + [`Lightning`](https://github.com/hipudding/pytorch-lightning/tree/npu_support)

**Other Dependencies**: see in `requirements.txt`

#### Datasets

- **Image Dataset**

We use Imagenet2012 as our Image dataset.
```
imagenet
└── train/
    ├── n01440764
        ├── n01440764_10026.JPEG
        ├── n01440764_10027.JPEG
        ├── ...
    ├── n01443537
    ├── ...
└── val/
    ├── ...
```

- **Video Dataset**

We use UCF-101 as our Video Dataset
```
UCF101
└── train/
    ├── class_0
        ├── video_1.mp4
        ├── video_2.mp4
        ├── ...
    ├── class_1
    ├── class_2
└── val/
    ├── ...
```
The preparation of UCF-101 can be referred to [VideoGPT](https://github.com/wilson1yan/VideoGPT)

- **Text2Image Datasets**

We recommend the data are organized in the following tar format.
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

### ⚡ Training & Evaluation
The training and evaluation scripts are in <a href="docs/Open-MAGVIT2.md">Open-MAGVIT2.md</a> and <a href="docs/IBQ.md">IBQ.md</a>.

## ❤️ Acknowledgement
We thank [Lijun Yu](https://me.lj-y.com/) for his encouraging discussions. We refer a lot from [VQGAN](https://github.com/CompVis/taming-transformers) and [MAGVIT](https://github.com/google-research/magvit). We also refer to [LlamaGen](https://github.com/FoundationVision/LlamaGen), [VAR](https://github.com/FoundationVision/VAR), [RQVAE](https://github.com/kakaobrain/rq-vae-transformer) and [VideoGPT](https://github.com/wilson1yan/VideoGPT), [OmniTokenizer](https://github.com/FoundationVision/OmniTokenizer). Thanks for their wonderful work.
