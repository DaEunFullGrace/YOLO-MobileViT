# YOLOv4+MobileViT

This is term project from DL_NLP_101 course
MobileViT block is implemented based on YOLOv4 architecture

You can see [vitblock] in cfg file
MobileViT block can be added by simply adding [vitblock] in the cfg file
```
[vitblock]
filters=512
size=3
d_model=32
num_heads=8
patch_size=4
depth=4
```
## How to fill [vitblock] field?
~Will be written~ 

## Result
Max epoch : 300
MobileViT block depth : 2
Patch size : 4/8
Training size : 512x512 (Original dataset size is 1280x1280)

### YOLOv4
Dataset : DOTA
||Baseline|Neck(patch size:4)|
|------|---|---|
|mAP:0.5|59.1%|64.8%|
|mAP:0.5-0.95|34.7%|38.77%|

Dataset : Vehicle from DOTA (small vehicle + large vehicle class)
||Baseline|Neck(patch size:4)|
|------|---|---|
|mAP:0.5|81.5%|79.4%|
|mAP:0.5-0.95|45.8%|48.4%|

### YOLOv4-Tiny
Dataset : DOTA
||Baseline|Neck(patch size:4)|Neck(patch size:8)|Backbone(patch size:4)|Backbone+Neck|
|------|---|---|---|---|---|
|mAP:0.5|41.93%|44.91%|43.72%|-|-|
|mAP:0.5-0.95|21.6%|23.86%|23.25%|-|-|

Dataset : Vehicle from DOTA (small vehicle + large vehicle class)
||Baseline|Neck(patch size:4)|Neck(patch size:8)|Backbone(patch size:4)|Backbone+Neck|
|------|---|---|---|---|---|
|mAP:0.5|61.69%|64.08%|63.05%|62.25%|64.4%|
|mAP:0.5-0.95|33.13%|35.42%|33.91%|35.94|33.13%|

## Acknowledgements

* [https://github.com/hugman/DL_NLP_101](https://github.com/hugman/DL_NLP_101)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4] (https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/chinhsuanwu/mobilevit-pytorch] (https://github.com/chinhsuanwu/mobilevit-pytorch)
