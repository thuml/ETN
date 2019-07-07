# Learning to Transfer Examples for Partial Domain Adaptation

Code release for  **[Learning to Transfer Examples for Partial Domain Adaptation (CVPR 2019)](https://youkaichao.github.io/files/cvpr2019/1855.pdf)** 

## Requirements
- python 3.6+
- PyTorch 1.0

`pip install -r requirements.txt`

## Usage

- download datasets

- write your config file

- `python main.py --config /path/to/your/config/yaml/file`

- train (configurations in `officehome-train-config.yaml` are only for officehome dataset):

  `python main.py --config officehome-train-config.yaml`

- test

  `python main.py --config officehome-test-config.yaml`
  
- monitor (tensorboard required)

  `tensorboard --logdir .`

## Checkpoints

We provide the checkpoints for officehome datasets at [Google Drive](https://drive.google.com/drive/folders/1jZsa1Bv3jByiFZrE9bMie9a9OBqT8yxb?usp=sharing).

## Citation
please cite:
```
@InProceedings{ETN_2019_CVPR,
author = {Zhangjie Cao, Kaichao You, Mingsheng Long, Jianmin Wang, Qiang Yang},
title = {Learning to Transfer Examples for Partial Domain Adaptation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

## Contact
- youkaichao@gmail.com
- caozhangjie14@gmail.com
- longmingsheng@gmail.com
