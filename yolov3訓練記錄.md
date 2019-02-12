# improved_yolov3的代码解读

## １．改进

- 将Convolutional + Batch-norm两层融合为1 层，神经网络性能提升百分之7。
-  使用`darknet detector demo`将网络在FullHD上的检测性能提升1.2倍，在4K上提升了2倍。
- `使用OpenCV SSE/AVX functions 代替原来的手写函数，将训练时的数据`扩增的性能提升了3.5倍。
- 当random = 1时，即使用多尺度的训练时优化内存分配。
- optimized initialization GPU for detection - we use batch=1 initially instead of re-init with batch=1
- **加入了准确率的计算 mAP, F1, IoU, Precision-Recall 使用命令 `darknet detector map`...**
- 训练时加入了 average-Loss and accuracy-mAP 图像的绘制(`-map` flag) 
- 在训练时加入了针对数据集的anchor的计算

## ２．如何使用

- Yolo v3 COCO - **image**: `darknet.exe detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights -i 0 -thresh 0.25 `**或者**darknet.exe detect cfg/yolov3.cfg yolov3.weights -i 0 -thresh 0.25`
- **输出图像中物体的坐标**: `darknet.exe detector test cfg/coco.data yolov3.cfg yolov3.weights -ext_output dog.jpg`
- Yolo v3 COCO - **测试视频**: `darknet.exe detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights -ext_output test.mp4`
- **Yolo v3 COCO -接入实时视频流** `darknet.exe detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights -c 0`
- Yolo v3 - **将视频的输出结果保存为 res.avi**: `darknet.exe detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights -thresh 0.25 test.mp4 -out_filename res.avi`
- **将train.txt中对应的图像检测并输出到的result.txt文件中 :**
  `darknet.exe detector test cfg/coco.data yolov3.cfg yolov3.weights -dont_show -ext_output < data/train.txt > result.txt`
- **计算数据集对应的anchors**: `darknet.exe detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416`
- **计算 mAP@IoU=50**: `darknet.exe detector map data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights`
- **计算 mAP@IoU=75**: `darknet.exe detector map data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights -iou_thresh 0.75`

- Yolo v3 **Tiny** COCO - video: `darknet.exe detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights test.mp4`
- Yolo v3 Tiny **on GPU #0**: `darknet.exe detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights -i 0 test.mp4`
- 186 MB Yolo9000 - image: `darknet.exe detector test cfg/combine9k.data yolo9000.cfg yolo9000.weights`

## ３．如何在linux 上编译

在darknet directory中执行make操作. 在make之前，需要在Makefile中执行进行一些参数上的设定:

- `GPU=1` 使用cuda加速
- `CUDNN=1` 
- `CUDNN_HALF=1` to build for Tensor Cores (on Titan V / Tesla V100 / DGX-2 and later　**会检测速度提升3倍，训练速度提升２倍。**
- `OPENCV=1` to build with OpenCV 3.x/2.4.x - **设置这个选项，后续才可以进行视频或者视频流的检测**
- `DEBUG=1` **可以调试 Yolo的版本**
- `OPENMP=1` **使用OpenMP支持构建，通过使用多核CPU来加速Yolo**
- `LIBSO=1` **构建一个库`darknet.so`和二进制可运行文件`uselib　**`　　Or you can try to run so `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib test.mp4` How to use this SO-library from your own code - you can look at C++ example: <https://github.com/AlexeyAB/darknet/blob/master/src/yolo_console_dll.cpp> or use in such a way: `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib data/coco.names cfg/yolov3.cfg yolov3.weights test.mp4`

## ４．如何训练 (Pascal VOC Data)

- 下载含有预训练参数的卷积网络 (154 MB): <http://pjreddie.com/media/files/darknet53.conv.74> 并将其放置到 `build\darknet\x64`目录下。
- 下载 The Pascal VOC Data 数据集，并 将其解压到目录build\darknet\x64\data\voc，则会产生目录build\darknet\x64\data\voc\VOCdevkit\

- 下载文件 `voc_label.py` 到目录　build\darknet\x64\data\voc`: <http://pjreddie.com/media/files/voc_label.py>

- 运行命令: `python build\darknet\x64\data\voc\voc_label.py` (来产生文件: 2007_test.txt, 2007_train.txt, 2007_val.txt, 2012_train.txt, 2012_val.txt)

- 运行命令 `type 2007_train.txt 2007_val.txt 2012_*.txt > train.txt`将文件中的值重定向到train.txt文件中。

- Set `batch=64` and `subdivisions=8` in the file `yolov3-voc.cfg`这个值可以根据显卡的情况设置，如果显卡显存不够的话，则设置的更小一点，这个值的含义是一次去64个图片，分８次，每次输入８张图片，讲损失累计，最后64张图片全部送完以后，进行反向传播更新参数。

  

  **训练期间`avg` (loss) 出现了Nan,则训练就会错误，如果其行出现了Nan,如果出现的比重小于百分子30，则一切正常，如果大于，则可能是数据集本身的问题，也可能是batch和subdivision设置过大的问题。**

## 5.如果在多GPU上进行训练

- 首先使用１个GPU训练1000代: `darknet.exe detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74`　
- 然后停止，使用训练了1000 代的权重文件 `/backup/yolov3-voc_1000.weights`， 采用 多gpu训练，最多４个: `darknet.exe detector train cfg/voc.data cfg/yolov3-voc.cfg /backup/yolov3-voc_1000.weights -gpus 0,1,2,3`　在多GPU情况下，学习率是`learning_rate = 0.00025` (i.e. learning_rate = 0.001 / GPUs)，也就是说在在设置学习率的时候，是对应的单个gpu,同时在这种情况下，cfg文件中的burn_in =和max_batches =也会增加4倍。即 使用burn_in = 4000而不是1000。

## ６.如何检测自己的数据(yolov3)

这里仅仅针对yolo-v3，如果要选择其他的旧版本的yolo(to train old Yolo v2 `yolov2-voc.cfg`, `yolov2-tiny-voc.cfg`, `yolo-voc.cfg`, `yolo-voc.2.0.cfg`, ... [click by the link](https://github.com/AlexeyAB/darknet/tree/47c7af1cea5bbdedf1184963355e6418cb8b1b4f#how-to-train-pascal-voc-data))

1. 创建文件`yolo-obj.cfg` 内容和`yolov3.cfg` 一样

   设置 [`batch=64`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L3)

   设置 [`subdivisions=8`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L4)

   将`classes=80` 改成检测类别数，在yolo的３个检测维度上，修改的范围如下：

   - https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L610>
   - <https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L696>
   - <https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L783>

   修改卷积层filters的个数，计算方法是:filters=(classes + 5)x3,修改的范围如下：

   - <https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L603>
   - <https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L689>
   - <https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L776>

2. 在目录`build\darknet\x64\data\`创建文件 `obj.names` , 里面输入要检测的物体的类别名字。

3. 在目录 `build\darknet\x64\data\`创建文件 `obj.data` , 包含以下内容：(where **classes = number of objects**):

   ```
   classes= 2
   train  = data/train.txt
   valid  = data/test.txt
   names = data/obj.names
   backup = backup/
   ```

4. 把物体的格式为(.jpg)的图片放入`build\darknet\x64\data\obj\`中。

5. 使用下面的这个软件为图片做标记:https://github.com/AlexeyAB/Yolo_mark>，并产生xml文件，它将为每个.jpg图像文件创建.txt文件 - 在同一目录中并使用相同的名称，其中.txt文件的内容为  `<object-class> <x> <y> <width> <height>`，

   其中:

   - `<object-class>` - 是一个从`0` to `(classes-1)`的整数

   - `<x_center> <y_center> <width> <height>` -相对于图像宽高的浮点数，范围在0-1．

   - 例如: `<x> = <absolute_x> / <image_width>` or `<height> = <absolute_height> / <image_height>`

   - 注意这里: `<x_center> <y_center>` - 便是矩形框的中心坐标，不是左上角的坐标。

   - 例如：For example for `img1.jpg` you will be created `img1.txt` containing:

     ```
     1 0.716797 0.395833 0.216406 0.147222
     0 0.687109 0.379167 0.255469 0.158333
     1 0.420312 0.395833 0.140625 0.166667
     ```

　  　6．创建 `train.txt` 文件在目录`build\darknet\x64\data\`, for example containing:

　　　　　　data/obj/img1.jpg
　　　　　　data/obj/img2.jpg
　　　　　　data/obj/img3.jpg

​         7.下载预训练卷积网络: <https://pjreddie.com/media/files/darknet53.conv.74>并放到目录`build\darknet\x64`。

​         8.开始训练：

​            　　darknet detector train data/obj.data yolo-obj.cfg darknet53.conv.74　

​            其中：

- `yolo-obj_last.weights`每隔100代保存在目录`build\darknet\x64\backup\` 　　　　　　

- `yolo-obj_xxxx.weights`每隔1000代保存在目录`build\darknet\x64\backup\` 
- 远程查看训练过程中mAP & Loss-chart的变化情况，在不适用GUI情况下可以远程查看 `darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -dont_show -mjpeg_port 8090 -map` then open URL `http://ip-address:8090` in Chrome/Firefox browser)
- 8.1. 如果需要每过4个epoch计算一下map (set `valid=valid.txt` or `train.txt` in `obj.data` file) ，运行以下命令: `darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -map`　　　　

​         9.训练结果：

　　　训练完成以后在 build\darknet\x64\backup\　获得最后的输出文件yolo-obj_final.weights

　　10.每训练100代可以停止，下次可以从该断点继续训练，例如训练了2000 代以后，可以停止训练，在目录 build\darknet\x64\backup\中将yolo-obj_2000.weights　粘贴到build\darknet\x64\中继续上次断点训练，使用如下命令：darknet.exe detector train data/obj.data yolo-obj.cfg yolo-obj_2000.weights。也可以在45000 代中获得结果，45000是认为设置的最大步数。

​       11.如果需要改变网络图像的输入，图像的大小必须是32的倍数。

　   12.训练完如果要检测的话，darknet.exe detector test data/obj.data yolo-obj.cfg yolo-obj_8000.weights

## ７．如何检测自己的数据(yolo-tiny)　

　　　除过以下几步，其余的和上述一样:

- 下载yolov3-tiny的训练参数，这步做检测用: <https://pjreddie.com/media/files/yolov3-tiny.weights>
- Get pre-trained weights `yolov3-tiny.conv.15` using command: `darknet.exe partial cfg/yolov3-tiny.cfg yolov3-tiny.weights yolov3-tiny.conv.15 15`
- Make your custom model `yolov3-tiny-obj.cfg` based on `cfg/yolov3-tiny_obj.cfg` instead of `yolov3.cfg`
- Start training: `darknet.exe detector train data/obj.data yolov3-tiny-obj.cfg yolov3-tiny.conv.15`
- **如果想使用其他的backbone网络， ([DenseNet201-Yolo](https://github.com/AlexeyAB/darknet/blob/master/build/darknet/x64/densenet201_yolo.cfg) or [ResNet50-Yolo](https://github.com/AlexeyAB/darknet/blob/master/build/darknet/x64/resnet50_yolo.cfg)), you can download and get pre-trained weights as showed in this file: <https://github.com/AlexeyAB/darknet/blob/master/build/darknet/x64/partial.cmd> 如果不是基于其他的模型，也可以不需要预训练权重，自己的backbone会采用随机初始化权重的方式。**

## ８．何时停止呢？　　

　　　通常情况下要进行不少于2000次的迭代，其他具体判断方法如下：

- 当average loss **0.xxxxxx avg** 过了很多代都没有变化时，可以考虑降低。

> Region Avg IOU: 0.798363, Class: 0.893232, Obj: 0.700808, No Obj: 0.004567, Avg Recall: 1.000000, count: 8 Region Avg IOU: 0.800677, Class: 0.892181, Obj: 0.701590, No Obj: 0.004574, Avg Recall: 1.000000, count: 8
>
> **9002**: 0.211667, **0.060730 avg**, 0.001000 rate, 3.868000 seconds, 576128 images Loaded: 0.000000 seconds

　　　**9002** - iteration number (number of batch)

　　　**0.060730 avg** - average loss (error) - **the lower, the better**

- 一旦训练停止，你应该从darknet \ build \ darknet \ x64 \ backup中获取一些最后的.weights文件并选择其中最好的，比如可以有以下判断方法：

  ![Overfitting](https://camo.githubusercontent.com/51af5be5cfa94b6d741c90d10a163b168bf9170e/68747470733a2f2f6873746f2e6f72672f66696c65732f3564632f3761652f3766612f35646337616537666164396434653365623361343834633538626663316666352e706e67)

有些时候网络过拟合了，最后迭代的结果不一定最好，所以就需要实施判断，并选择最好的结果。

1. 首先在文件`obj.data` 必须确定表示验证集图片路径的文件 `valid = valid.txt` ，如果没有验证集图片, just copy `data\train.txt` to `data\valid.txt`.　　

2. 如果训练到9000停止，可以使用如下命令验证之前的一些权重文件的检测结果，然后比较输出结果：

   - darknet.exe detector map data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights`
   - `darknet.exe detector map data/obj.data yolo-obj.cfg backup\yolo-obj_8000.weights`
   - `darknet.exe detector map data/obj.data yolo-obj.cfg backup\yolo-obj_9000.weights`

   在输出结果中，选择map或者iou最大的。

3. 或者在训练的过程中使用-map 选项。

   darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -map

   会得到如下的训练图. mAP 每４个epoch计算一次，using `valid=valid.txt` file that is specified in `obj.data` file (`1 Epoch = images_in_train_txt / batch` iterations)

   ![loss_chart_map_chart](https://camo.githubusercontent.com/d60dfdba6c007a5df888747c2c03664c91c12c1e/68747470733a2f2f6873746f2e6f72672f776562742f79642f766c2f61672f7964766c616775746f66327a636e6a6f64737467726f656e3861632e6a706567)

   mAP是PascalVOC竞赛中的默认精度度量，这与MS COCO竞赛中的AP50指标相同。 就Wiki而言，指标Precision和Recall与PascalVOC竞争中的含义略有不同，但IoU总是具有相同的含义。

   

   ![precision_recall_iou](https://camo.githubusercontent.com/ffd00e8c7f54d4710edea3bb47e201c8bedab074/68747470733a2f2f6873746f2e6f72672f66696c65732f6361382f3836362f6437362f63613838363664373666623834303232383934306462663434326137663036612e6a7067)















