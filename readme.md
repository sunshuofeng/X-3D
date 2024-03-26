# X-3D
This is a official implementation of X-3D proposed by our paper X-3D: Explicit 3D Structure Modeling for Point Cloud Recognition (CVPR 2024).


## News
- *Feb, 2024:* X-3D accepted by CVPR 2024.
- *Mar, 2024:* Code Release.

## Install
```
source install.sh
```
Please change the version of the relevant package in `install.sh` according to your CUDA version.

## Dataset
Please refer to [PointNeXt tutorial](https://guochengqian.github.io/PointNeXt/) to download the datasets. 

## Usage & Model Zoo
### S3DIS 


|       Model      |       [Area-5 mIoU/OA ](https://drive.google.com/drive/folders/1syPHn5ivcOGhyc7tO_0MCjw7ZDIvFg4f)      |     [6-fold mIoU/OA](https://drive.google.com/drive/folders/1asb-XmxUe2DoWH-q5eKFmk3gmVr-ytbb)     | Params (M) | FLOPs (G) |
| :--------------: | :----------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------: | :--------: | :-------: |
|  PointMetaBase-L | [69.723/90.702](https://drive.google.com/file/d/1LRMu_JasWhlM9tFumMouUPeDeTnD8Mg3/view?usp=share_link) | [75.6/90.6](https://drive.google.com/file/d/15T2RxPSW8mmvqHTHF7meXPs-yslPY_nX/view?usp=share_link) |     2.7    |    2.0    |
|       +X-3D      |  [71.8/91.2](https://drive.google.com/drive/folders/15C90VveHP_5LnrOmKYhTFvXQg0TFk21E?usp=drive_link)  |                                              76.7/91.1                                             |     3.8    |    2.2    |
| PointMetaBase-XL | [71.597/90.551](https://drive.google.com/file/d/1zuaLaKLoFF8r3y0ano04tRm_FoLSaZ8N/view?usp=share_link) | [76.3/91.0](https://drive.google.com/file/d/123usjeRwr5HUMCryis2soE0er7eUU7dR/view?usp=share_link) |    15.3    |    9.2    |
|       +X-3D      |       [72.1/91.4](https://drive.google.com/drive/folders/1OR9bsE0keHvwYCu7J45Fz_ZVlUazppCL?usp=drive_link)                                                                                               |                             77.7/91.6                                                                       |         18.1   | 9.8          |


#### Train
```
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointmetabase-l-x-3d.yaml wandb.use_wandb=True
```
#### Test
```
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointmetabase-l-x-3d.yaml wandb.use_wandb=False mode=test --pretrained_path path/to/pretrained/model
```

#### Profile Parameters, FLOPs, and Throughput
```
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/s3dis/pointmetabase-l-x-3d.yaml batch_size=16 num_points=15000 flops=True timing=True
```


### ScanNet
|       Model      |      [Val mIoU](https://drive.google.com/drive/folders/1k-vTW8GWB4XLYVrO1ZQ5UwjfkodxjckE)     | Test mIoU | Params (M) | FLOPs (G) | TP (ins./sec.) |
| :--------------: | :-------------------------------------------------------------------------------------------: | :-------: | :--------: | :-------: | :------------: |
|  PointMetaBase-L | [71.0](https://drive.google.com/file/d/1RooGlWAvcddGa8N9i5y5iBRB8zAmkzpb/view?usp=share_link) |     -     |     2.7    |    2.0    |       187      |
|       +X-3D      |                                             [71.8](https://drive.google.com/drive/folders/15YsTqV52WZYAh1fKkiE_XpZhe5icCuFb?usp=drive_link)                                             |           |     3.8    |    2.2    |                |
| PointMetaBase-XL | [71.8](https://drive.google.com/file/d/1HYR5bZnpDAMd8XaPMJKFeuhgYn3wnuPv/view?usp=share_link) |     -     |    15.3    |    9.2    |       104      |
|       +X-3D      |                                              [72.8](https://drive.google.com/drive/folders/1TUab8Z5E-pItVL-YUQdnbM2ISpNmLM6h?usp=drive_link)                                             |           |    18.1    |    9.8    |                |
#### Train
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/segmentation/main.py --cfg cfgs/scannet/pointmetabase-l-x3d.yaml wandb.use_wandb=True
```
#### Test
```
CUDA_VISIBLE_DEVICES=0  python examples/segmentation/main.py --cfg cfgs/scannet/pointmetabase-l-x3d.yaml mode=test dataset.test.split=val --pretrained_path path/to/pretrained/model
```
#### Profile Parameters, FLOPs, and Throughput
```
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/scannet/pointmetabase-l-x3d.yaml batch_size=16 num_points=15000 flops=True timing=True
```




## Acknowledgment
This repository is built on reusing codes of [OpenPoints](https://github.com/guochengqian/openpoints)， [PointNeXt](https://github.com/guochengqian/PointNeXt) and [PointMetaBase](https://github.com/linhaojia13/PointMetaBase.git)