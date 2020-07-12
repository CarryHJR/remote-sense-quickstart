## 数据
第一步：获取容器镜像。在安装好Docker的机器上执行下面命令，获取对应的数据：
```
docker pull 2020gaofen/data:automatic_bridge_detection_in_optical_images_2020
```
第二步：将数据导出到电脑硬盘。参赛数据在容器的/data路径下，执行下面命令将数据拷贝到电脑硬盘中，下面是将数据导出到硬盘
的/mnt/share/data文件夹下：
```
docker run -it -v /mnt/share/data:/test 2020gaofen/data:automatic_bridge_detection_in_optical_images_2020 /bin/sh
cp -r /data/ /test
```
## 预处理
这个xml格式有点特别

![](https://upload-images.jianshu.io/upload_images/23853026-daa863aea9e9f031.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/400)
和voc的不一样， 不过看起来就是四个框的角点，找最大最小值应该就是xmin ymin xmax ymax了，随手转成coco格式

图像是tif后缀，我读取了就是普通的光学影像, PIL就可以读取

![](https://upload-images.jianshu.io/upload_images/23853026-8f2f3851788a2dc0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/300)

这部分的代码在[https://github.com/CarryHJR/remote-sense-quickstart/blob/master/bridge-detection/preprocess.ipynb](https://github.com/CarryHJR/remote-sense-quickstart/blob/master/bridge-detection/preprocess.ipynb)

## train
我用的mmdetection 1.2.0，可谓1.x中最稳定版本
```
curl https://codeload.github.com/open-mmlab/mmdetection/tar.gz/v1.2.0 -o v1.2.tar.gz 
```
解压后日常安装mmcv(0.5.5) pycocotools 以及`python setup.py develop`

既然数据集做好了，那就搞个cascade r50试试水吧, 改个num_class和数据集位置就好了，最后的config放在了
[https://github.com/CarryHJR/remote-sense-quickstart/blob/master/bridge-detection/config.py](https://github.com/CarryHJR/remote-sense-quickstart/blob/master/bridge-detection/config.py)


## eval
日志文件在 [https://github.com/CarryHJR/remote-sense-quickstart/blob/master/bridge-detection/20200712_014048.log.json](https://github.com/CarryHJR/remote-sense-quickstart/blob/master/bridge-detection/20200712_014048.log.json)

![](https://upload-images.jianshu.io/upload_images/23853026-04088ad41c2be0bf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

mmdet的log里val map50 74.3 问题不大，可视化一下

![](https://upload-images.jianshu.io/upload_images/23853026-e39143801dc46f7c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## think
按照经验 cascade r50 74.3，加上mixup mosaic stitcher dcn cbam + senet + all data + ensemble 90map应该问题不大
