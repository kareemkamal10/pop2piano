# Pop2Piano : Pop Audio-based Piano Cover Generation
---
**التحديثات والإصدارات (2025–2026):** المشروع محدّث ليعمل مع آخر إصدارات المكتبات. للتفاصيل: [UPGRADE_AND_VERSIONS.md](UPGRADE_AND_VERSIONS.md).

##
- [Paper](https://arxiv.org/abs/2211.00895)
- [Project Page](http://sweetcocoa.github.io/pop2piano_samples)

### Training (Upgraded 2026)
- [Train on Colab](https://colab.research.google.com/github/kareemkamal10/pop2piano/blob/upgradeProject/train.ipynb) ✨ **New Features:**
  - Mixed Precision (FP16) - 2x faster
  - Arabic Maqamat Support (راست، حجاز، بياتي...)
  - Piano Rules - Better playability
  - Resume from Checkpoint

### Inference (Use Pre-trained Model)
- [Inference on Colab](https://colab.research.google.com/github/kareemkamal10/pop2piano/blob/main/inference_colab.ipynb)
- [Inference on Kaggle](https://kaggle.com/kernels/welcome?src=https://github.com/kareemkamal10/pop2piano/blob/main/inference_kaggle.ipynb)

## How to prepare dataset
### Download Original Media
---
- List of data : ```train_dataset.csv```
- Downloader : ```download/download.py```
    - ```python download.py ../train_dataset.csv output_dir/```

### Preprocess Data
---
- [Details](./preprocess/)







