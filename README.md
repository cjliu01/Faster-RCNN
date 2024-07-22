# Faster-RCNN
Implementation of MultiScaleRoIAlign and MultiScaleRoIAlign in Faster RCNN, code mainly comes from [deep-learning-for-image-processing/pytorch_object_detection/faster_rcnn](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/faster_rcnn).

### Prepare environment：

* Python3.6/3.7/3.8
* Pytorch1.7.1
* pycocotools
* Ubuntu or CentOS
* detailed environment config refer to `requirements.txt`

### File structure：

```
  ├── backbone: backbone files
  ├── network_files: Faster R-CNN framework files
  ├── train_utils: training and validation modules
  ├── my_dataset.py: custom dataset for reading VOC dataset
  ├── train_mobilenet.py: using MobileNetV2 as backbone
  ├── train_resnet50_fpn.py: using resnet50+FPN as backbone
  ├── train_multi_GPU.py: using multiGPU for training
  ├── predict.py: prediction script
  ├── validation.py: validation script and generated record_mAP.txt
  └── pascal_voc_classes.json: pascal_voc label json
```

### Download pre-trained weights（place it in the `backbone` folder）：

* MobileNetV2 weights download link: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
* Resnet50 weights download link: https://download.pytorch.org/models/resnet50-0676ba61.pth
* ResNet50+FPN weights download link: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

**Note** that the pretraining weights downloaded should be renamed, for example, the downloaded weight is named `fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` and should be renamed  `fasterrcnn_resnet50_fpn_coco.pth`.

### prepare datasets

The dataset used in this repo is Pascal VOC2012 train/val. Dataset download link at [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).

### training process

* ensure that the dataset is ready

* ensure that the pretraining weights are downloaded properly

* To using **mobilenetv2** as backbone, directly use the `train_mobilenet.py`

* To using **resnet50_fpn** as backbone, directly use the `train_res50_fpn.py`

* If you want to use multiGPU training, use the following command

  ```python
  CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 train_multi_GPU.py --data-path [PASCAL_VOC_2012 datasets path] -b [batch_size]
  ```

  

  

  `

