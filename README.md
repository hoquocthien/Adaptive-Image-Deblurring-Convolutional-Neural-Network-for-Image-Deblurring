# Adaptive-Image-Deblurring-Convolutional-Neural-Network-for-Image-Deblurring

Quoc-Thien Ho, Minh-Thien Duong, Seongsoo Lee and Min-Cheol Hong
> Abstract: Motion blur is a complex phenomenon caused by the relative movement between an observed object and an imaging sensor during the exposure time, resulting in degradation in the image quality. Deep learning-based methods, particularly convolutional neural networks (CNNs), have shown promise in motion deblurring. However, the limited receptive fields due to the small kernel sizes of CNNs limit their ability to achieve optimal performance. Moreover, supervised deep learning-based deblurring methods often exhibit overfitting in their training datasets. Models trained on widely used synthetic blur datasets frequently fail to generalize in other blur domains in real-world scenarios and often produce undesired artifacts. To address these challenges, we propose the Spatial Feature Selection Network (SFSNet), which incorporates a Regional Feature Extractor (RFE) module to expand the receptive field and effectively select critical spatial features in order to improve the deblurring performance. In addition, we present the BlurMix dataset, which includes diverse blur types, as well as a meta-tuning strategy for effective blur domain adaptation. Our meta-tuning strategy enables the network to rapidly adapt to novel blur distributions with minimal additional training, and mitigate overfitting. The experimental results show that the meta-tuning variant of the SFSNet eliminates unwanted artifacts and significantly improves the deblurring performance across various blur domains.

## Installation 
This project is built with PyTorch 3.12, Pytorch 2.5.1, CUDA 12.4.

For installing, follow these instructions:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install natsort opencv-python einops ptflops lmdb tqdm scikit-image warmup_scheduler
```

## Dataset 
- Download deblur dataset: [GoPro](https://seungjunnah.github.io/Datasets/gopro.html), [HIDE](https://github.com/joanshen0508/HA_deblur?tab=readme-ov-file), [REDS](https://seungjunnah.github.io/Datasets/reds.html), [RealBlur](https://cg.postech.ac.kr/research/realblur/), [RSBlur](https://cg.postech.ac.kr/research/rsblur/), [ReLoBlur](https://leiali.github.io/ReLoBlur_homepage/index.html).

- - Preprocess data folder. The data folder should be like the format:
  
├─ test  ## GoPro/ HIDE / REDS / RealBlur / RSBlur /ReLoBlur.

│ ├─ input      &emsp;&emsp; 

│ │ ├─ xxxx.png

│ │ ├─ ......

│ │

│ ├─ target

│ │ ├─ xxxx.png

│ │ ├─ ......

│
├─ train   ## GOPRO / Blur Mix

│ ├─ ...... (same as test)
