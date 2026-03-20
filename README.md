
<a id="top"></a>
<div align="center">
 <img src="./assets/logo.png" width="500"> 
  <h1>(AAAI 2026) HABIT: Chrono-Synergia Robust Progressive Learning Framework for Composed Image Retrieval</h1>
  
  <p>
      <a href="https://aaai.org/Conferences/AAAI-26/"><img src="https://img.shields.io/badge/AAAI-2026-blue.svg?style=flat-square" alt="AAAI 2026"></a>
      <a href="https://arxiv.org/abs/coming soon"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-Coming.Soon-b31b1b.svg"></a>
      <a href="https://ojs.aaai.org/index.php/AAAI/article/view/37608"><img alt='Paper' src="https://img.shields.io/badge/Paper-green.svg"></a>
    <a href="https://lee-zixu.github.io/HABIT.github.io/"><img alt='page' src="https://img.shields.io/badge/Website-orange"></a>
    <a href="https://lee-zixu.github.io"><img src="https://img.shields.io/badge/Author Page-blue.svg" alt="Author Page"></a>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?&logo=pytorch&logoColor=white"></a>
    <img src="https://img.shields.io/badge/python-≥3.8-blue?style=flat-square" alt="Python">
    <a href="https://github.com/"><img alt='stars' src="https://img.shields.io/github/stars/lee-zixu/HABIT?style=social"></a>
    <!-- <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"> -->
  </p>

  <p>
    <b>Accepted by AAAI 2026:</b> A robust progressive learning framework tackling the Noise Triplet Correspondence (NTC) problem in Composed Image Retrieval (CIR).
  </p>
</div>


## 📌 Introduction

**HABIT** (cHrono-synergiA roBust progressIve learning framework for composed image reTrieval) is our proposed robust learning framework for Composed Image Retrieval, accepted by AAAI 2026. Based on an in-depth analysis of the "Noisy Triplet Correspondence (NTC)" problem in real-world retrieval scenarios, HABIT effectively addresses the shortcomings of existing methods in precisely estimating composed semantic discrepancies and progressively adapting to modification discrepancies.

[⬆ Back to top](#top)

## 📢 News
- **[2026-03-18]** 🚀 Released all codes for HABIT.
- **[2025-11-08]** 🔥 Our paper *"HABIT: Chrono-Synergia Robust Progressive Learning Framework for Composed Image Retrieval"* has been accepted by **AAAI 2026**!

[⬆ Back to top](#top)

## ✨ Key Features

  - 🧠 **Mutual Knowledge Estimation (MKE)**: Precisely quantifies sample cleanliness by computing the *Transition Rate* of mutual knowledge between composed features and target images, effectively identifying clean samples that align with modification semantics.
  - ⏳ **Dual-consistency Progressive Learning (DPL)**: Introduces a collaborative mechanism between historical and current models to simulate human habit formation (retaining good habits and calibrating bad ones), enabling robust learning against noisy data interference.
  - 🛡️ **Highly Robust to NTC**: Maintains State-of-the-Art (SOTA) retrieval performance under various Noise Triplet Correspondence (NTC) settings with different noise ratios (0%, 20%, 50%, 80%).

[⬆ Back to top](#top)

## 🏗️ Architecture

<p align="center">
  <img src="assets/HABIT-AAAI26.png" alt="HABIT architecture" width="1000">
  <figcaption><strong>Figure 1.</strong> HABIT consists of two modules: (a) Mutual Knowledge Estimation and (b) Dual-consistency Progressive Learning. </figcaption>
</p>

[⬆ Back to top](#top)

## 🏃‍♂️ Experiment-Results

### CIR Task Performance

> 💡 <span style="color:#2980b9;">**Note for Fully-Supervised CIR Benchmarking:**</span> <br>
> 🎯 The **0% noise** setting in the table below is equivalent to the **traditional fully-supervised CIR** paradigm. We highlight this `0%` block to facilitate direct and fair comparisons for researchers working on conventional supervised methods.


#### FIQ:

<caption><strong>Table 1.</strong> Performance comparison on FashionIQ in terms of R@K (%). The best result under each noise ratio is highlighted in
bold, while the second-best result is underlined.</caption>

![](./assets/results-fiq.png)



#### CIRR：
<caption><strong>Table 2.</strong> Performance comparison on the CIRR test set in terms of R@K (%) and Rsub@K (%). The best and second-best results
are highlighted in bold and underlined, respectively.</caption>

![](./assets/results-cirr.png)

[⬆ Back to top](#top)

---


## Table of Contents

- [Introduction](#-introduction)
- [News](#-news)
- [Key Features](#-key-features)
- [Architecture](#️-architecture)
- [Experiment Results](#️-experiment-results)
- [Install](#-install)
- [Data Preparation](#-data-preparation)
- [Quick Start](#-quick-start)
  - [Training under Noisy Settings](#1-training-under-noisy-settings)
  - [Testing](#2-testing)
- [Project Structure](#-project-structure)
- [Acknowledgement](#-acknowledgement)
- [Contact](#️-contact)
- [Citation](#️-citation)
- [Support & Contributing](#-support--contributing)

---


## 📦 Install

**1. Clone the repository**

```bash
git clone https://github.com/Lee-zixu/HABIT
cd HABIT
```

**2. Setup Python Environment**

The code is evaluated on **Python 3.8.10** and **CUDA 12.6**. We recommend using Anaconda to create an isolated virtual environment:

```bash
conda create -n habit python=3.8
conda activate habit

# Install PyTorch (The evaluated environment uses Torch 2.1.0 with CUDA 12.1 compatibility)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install core dependencies
pip install open-clip-torch==2.24.0 scikit-learn==1.3.2 transformers==4.25.0 salesforce-lavis==1.0.2 timm==0.9.16
```

> **Note**: Key dependencies include `salesforce-lavis` for the base architecture, `open-clip-torch` for vision-language features, and `scikit-learn` for DBSCAN clustering during Noise Discrimination.

[⬆ Back to top](#top)

-----

## 📂 Data Preparation

We evaluated our framework on two standard datasets: [FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq) and [CIRR](https://github.com/Cuberick-Orion/CIRR). Please download the datasets first.

<details>
<summary><b>Click to expand: FashionIQ Dataset Directory Structure</b></summary>

Please follow the official instructions to download the FashionIQ dataset. Once downloaded, ensure the folder structure looks like this:

```text
├── FashionIQ
│   ├── captions
│   │   ├── cap.dress.[train | val | test].json
│   │   ├── cap.toptee.[train | val | test].json
│   │   ├── cap.shirt.[train | val | test].json
│   ├── image_splits
│   │   ├── split.dress.[train | val | test].json
│   │   ├── split.toptee.[train | val | test].json
│   │   ├── split.shirt.[train | val | test].json
│   ├── dress
│   │   ├── [B000ALGQSY.jpg | B000AY2892.jpg | B000AYI3L4.jpg |...]
│   ├── shirt
│   │   ├── [B00006M009.jpg | B00006M00B.jpg | B00006M6IH.jpg | ...]
│   ├── toptee
│   │   ├── [B0000DZQD6.jpg | B000A33FTU.jpg | B000AS2OVA.jpg | ...]
```

</details>

<details>
<summary><b>Click to expand: CIRR Dataset Directory Structure</b></summary>

Please follow the official instructions to download the CIRR dataset. Once downloaded, ensure the folder structure looks like this:

```text
├── CIRR
│   ├── train
│   │   ├── [0 | 1 | 2 | ...]
│   │   │   ├── [train-10108-0-img0.png | train-10108-0-img1.png | ...]
│   ├── dev
│   │   ├── [dev-0-0-img0.png | dev-0-0-img1.png | ...]
│   ├── test1
│   │   ├── [test1-0-0-img0.png | test1-0-0-img1.png | ...]
│   ├── cirr
│   ├── captions
│   │   ├── cap.rc2.[train | val | test1].json
│   ├── image_splits
│   │   ├── split.rc2.[train | val | test1].json
```

</details>

[⬆ Back to top](#top)

-----

## 🚀 Quick Start

### 1\. Training under Noisy Settings

In our implementation, we introduce the `noise_ratio` parameter to simulate varying degrees of NTC (Noise Triplet Correspondence) interference. You can reproduce the experimental results from the paper by modifying the `--noise_ratio` parameter (default options evaluated are `0.0`, `0.2`, `0.5`, `0.8`).

**Training on FashionIQ:**

```bash
python train.py \
    --dataset fashioniq \
    --fashioniq_path "/path/to/FashionIQ/" \
    --model_dir "./checkpoints/fashioniq_noise0.2" \
    --noise_ratio 0.2 \
    --batch_size 256 \
    --num_epochs 20 \
    --lr 2e-5
```

**Training on CIRR:**

```bash
python train.py \
    --dataset cirr \
    --cirr_path "/path/to/CIRR/" \
    --model_dir "./checkpoints/cirr_noise0.5" \
    --noise_ratio 0.5 \
    --batch_size 256 \
    --num_epochs 20 \
    --lr 2e-5
```

> **💡 Tips:** > - Our model is based on the powerful BLIP-2 architecture. It is highly recommended to run the training on GPUs with sufficient memory (e.g., NVIDIA A40 48G / V100 32G).
>
>   - The best model weights and evaluation metrics generated during training will be automatically saved in the `best_model.pt` and `metrics_best.json` files within your specified `--model_dir`.

### 2\. Testing

To generate the prediction files on the CIRR dataset for submission to the [CIRR Evaluation Server](https://cirr.cecs.anu.edu.au/), run the following command:

```bash
python src/cirr_test_submission.py checkpoints/cirr_noise0.5/
```

*(The corresponding script will automatically output `.json` based on the generated best checkpoints in the folder for online evaluation.)*

[⬆ Back to top](#top)

-----

## 📁 Project Structure

Our code is deeply customized based on the LAVIS framework. The core implementations are centralized in the following files:

```text
HABIT/
├── lavis/
│   ├── models/
│   │   └── blip2_models/
│   │       └── HABIT.py      # 🧠 Core model implementation: Includes MKE, DPL modules, and loss functions
├── train.py                  # 🚀 Training entry point: Controls noise_ratio injection and training loops
├── datasets.py 
├── test.py 
├── utils.py 
├── data_utils.py 
├── cirr_test_submission.py   # Auxiliary scripts
├── datasets/                 # Dataset loading and processing logic
└── README.md
```

-----





## 🤝 Acknowledgement

The implementation of this project references the [LAVIS](https://github.com/salesforce/LAVIS) framework and the noise setting concepts from [TME](https://github.com/li-shuxian/TME). We express our sincere gratitude to these open-source contributions\!

[⬆ Back to top](#top)


## ✉️ Contact
For any questions, issues, or feedback, please open an [issue](https://github.com/Lee-zixu/HABIT/issues) on GitHub or reach out to me at lizixu.cs@gmail.com.

[⬆ Back to top](#top)


## 🔗 Related Projects

*Ecosystem & Other Works from our Team*
<table style="width:100%; border:none; text-align:center; background-color:transparent;">
  <tr style="border:none;">
    <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/retrack-logo.png" alt="ReTrack" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>ReTrack (AAAI'26)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://lee-zixu.github.io/ReTrack.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/Lee-zixu/ReTrack" target="_blank">Code</a> | 
        <a href="https://ojs.aaai.org/index.php/AAAI/article/view/39507" target="_blank">Paper</a>
      </span>
    </td>
    <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/intent-logo.png" alt="INTENT" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>INTENT (AAAI'26)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://zivchen-ty.github.io/INTENT.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/ZivChen-Ty/INTENT" target="_blank">Code</a> | 
        <a href="https://ojs.aaai.org/index.php/AAAI/article/view/39181" target="_blank">Paper</a>
      </span>
    </td>  
    <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/hud-logo.png" alt="HUD" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>HUD (ACM MM'25)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://zivchen-ty.github.io/HUD.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/ZivChen-Ty/HUD" target="_blank">Code</a> | 
        <a href="https://dl.acm.org/doi/10.1145/3746027.3755445" target="_blank">Paper</a>
      </span>
    </td>
    <!-- </tr>
  <tr style="border:none;"> -->
    <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/offset-logo.png" alt="OFFSET" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>OFFSET <br>(ACM MM'25)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://zivchen-ty.github.io/OFFSET.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/ZivChen-Ty/OFFSET" target="_blank">Code</a> | 
        <a href="https://dl.acm.org/doi/10.1145/3746027.3755366" target="_blank">Paper</a>
      </span>
    </td>
    <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/encoder-logo.png" alt="ENCODER" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>ENCODER (AAAI'25)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://sdu-l.github.io/ENCODER.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/Lee-zixu/ENCODER" target="_blank">Code</a> | 
        <a href="https://ojs.aaai.org/index.php/AAAI/article/view/32541" target="_blank">Paper</a>
      </span>
    </td>
  </tr>
</table>




## 📝⭐️ Citation

If you find our work or this code useful in your research, please consider leaving a **Star**⭐️ or **Citing**📝 our paper 🥰. Your support is our greatest motivation!

```bibtex
@inproceedings{HABIT,
  title={HABIT: Chrono-Synergia Robust Progressive Learning Framework for Composed Image Retrieval},
  author={Li, Zixu and Hu, Yupeng and Chen, Zhiwei and Zhang, Shiqi and Huang, Qinlei and Fu, Zhiheng and Wei, Yinwei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

[⬆ Back to top](#top)

## 🫡 Support & Contributing

We welcome all forms of contributions! If you have any questions, ideas, or find a bug, please feel free to:
- Open an [Issue](https://github.com/Lee-zixu/HABIT/issues) for discussions or bug reports.
- Submit a [Pull Request](https://github.com/Lee-zixu/HABIT/pulls) to improve the codebase.

[⬆ Back to top](#top)

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="500" alt="HABIT Demo">

  <br><br>

  <a href="https://github.com/Lee-zixu/HABIT">
    <img src="https://img.shields.io/badge/⭐_Star_US-000000?style=for-the-badge&logo=github&logoColor=00D9FF" alt="Star">
  </a>
  <a href="https://github.com/Lee-zixu/HABIT/issues">
    <img src="https://img.shields.io/badge/🐛_Report_Issues-000000?style=for-the-badge&logo=github&logoColor=FF6B6B" alt="Issues">
  </a>
  <a href="https://github.com/Lee-zixu/HABIT/pulls">
    <img src="https://img.shields.io/badge/🧐_Pull_Requests-000000?style=for-the-badge&logo=github&logoColor=4ECDC4" alt="Pull Request">
  </a>

  <br><br>
<a href="https://github.com/Lee-zixu/HABIT">
    <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=00D9FF&center=true&vCenter=true&width=500&lines=Thank+you+for+visiting+HABIT!;Looking+forward+to+your+attention!" alt="Typing SVG">
  </a>
</div>
