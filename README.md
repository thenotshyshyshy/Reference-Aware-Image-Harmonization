## Reference-Aware Image Harmonization



This is the official code of the Neural Networks 2025 paper: Reference-Aware Image Harmonization.


## Preparation
### 1. Clone this repo:
```bash
git clone https://github.com/chenhaoxing/HDNet
cd HDNet
```

### 2. Requirements
* Both Linux and Windows are supported, but Linux is recommended for compatibility reasons.
* We have tested on PyTorch 1.8.1+cu11. 

install the required packages using pip: 
```bash
pip3 install -r requirement.txt
```
or conda:
```bash
conda create -n rainnet python=3.8
conda activate rainnet
pip install -r requirement.txt
```
### 3. Prepare the data
Download [iHarmony4](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4) dataset in dataset folder and run  `data/preprocess_iharmony4.py` to resize the images (eg, 512x512, or 256x256) and save the resized images in your local device. 

### Training and validation
We provide the code in train_evaluate.py, which supports the model training, evaluation and results saving in iHarmony4 dataset.
```python
python train_evaluate.py --dataset_root <DATA_DIR> --save_dir results --batch_size 12 --device cuda 
```


## Citing RANet
If you use RANet in your research, please use the following BibTeX entry.

```BibTeX
@article{guo2025reference,
  title={Reference-Aware Image Harmonization},
  author={Guo, Han and Gu, Hongling and Zheng, Bolun and Zhang, Qianyu and Wang, Canjin and Wang, Yayun and Li, Zongpeng},
  journal={Neural Networks},
  pages={108439},
  year={2025},
  publisher={Elsevier}
}
```


## Acknowledgement
Many thanks to the nice work of  [RainNet](https://github.com/junleen/RainNet) and [HDNet](https://github.com/chenhaoxing/HDNet). Our codes and configs follow [HDNet](https://github.com/chenhaoxing/HDNet).

## Contacts
Please feel free to contact us if you have any problems. 

Email: [255060123@hdu.edu.cn]
