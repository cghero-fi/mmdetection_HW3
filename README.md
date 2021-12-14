# mmdetection_HW3

Data preparing
-------------

if you want to demo here, you can go to:
https://colab.research.google.com/drive/1ILo1csE8EUZLj4MoWExesMwJ0vWWrAdB?usp=sharing

####installation
```bash
!pip install cpython
!pip install git+https://github.com/waspinator/pycococreator.git
```
data preparing

```bash
$ git clone https://github.com/cghero-fi/mmdetection_HW3.git
!python mmdetection_HW3/mask2coco.py
```

if you want to use on local you need to change the 'dataset_train_path=' in mask2coco.py

you can get a train.json file in dictionary dataset/train/  for mmdetection

Training
-------------

if you want to demo here, you can go to here:
