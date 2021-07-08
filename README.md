# CanonPose

This is the original implementation of the CVPR 2021 Paper "CanonPose: Self-Supervised Monocular 3D Human Pose Estimation in the Wild" by Bastian Wandt, Marco Rudolph, Petrissa Zell, Helge Rhodin, and Bodo Rosenhahn.

## Getting Started

Required packages:

* pytorch
* pytorch3d

To install pytorch3d please follow the instructions at https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md

##  Data

Download the Alphapose (https://github.com/MVIG-SJTU/AlphaPose) detections and the pretrained morphing network from Google Drive:
https://drive.google.com/file/d/1gH5D-RKvdOLC-Cokuk4gaW9g5tAahQ1v/view?usp=sharing

Unpack the two folders data/ and models/ to the root folder of the project.

## Training

Training can be started with

```
python train.py
```


Unfortunately, due to licensing it is not possible to provide any data from Human3.6M. However, for training you only need 2D detections from your favorite 2D detector. Our Alphapose detections can be found in the archive above.

## Citation
Please cite the paper in your publications if it helps your research:

```
@inproceedings{Wandt2021Canonpose,
  author = {Wandt, Bastian and Rudolph, Marco and Zell, Petrissa and Rhodin, Helge and Rosenhahn, Bodo},
  booktitle = {Computer Vision and Pattern Recognition (CVPR)},
  month = jun,
  title = {CanonPose: Self-Supervised Monocular 3D Human Pose Estimation in the Wild},
  year = 2021
}
```

Links to the paper:

https://arxiv.org/pdf/2011.14679

## License

This project is licensed under the MIT License.
