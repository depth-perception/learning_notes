 

 

#  mix-up

 

在讲检测之前，我们先看看mixup在图像分类是怎么用的。mixup 源于顶会ICLR的一篇论文 [mixup: Beyond Empirical Risk Minimization](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1710.09412)，可以在**几乎无额外计算开销的情况下稳定提升1个百分点的图像分类精度**。当前mixup主要是用于图像分类，有两种主流的实现方式，我参考的是这个版本的代码：https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch/blob/master/Residual-Attention-Network/train_mixup.py

1. 对于输入的一个batch的待测图片images，我们将其和随机抽取的图片进行融合，融合比例为lam，得到混合张量inputs；
2. 第1步中图片融合的比例lam是[0,1]之间的随机实数，符合beta分布，相加时两张图对应的每个像素值直接相加，即 inputs = lam*images + (1-lam)*images_random；
3. 将1中得到的混合张量inputs传递给model得到输出张量outpus，随后计算损失函数时，我们针对两个图片的标签分别计算损失函数，然后按照比例lam进行损失函数的加权求和，即loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)；

我觉得这个过程不是很好讲清楚，我们直接看PyTorch版实现代码，代码很好理解：

```python
for i,(images,target) in enumerate(train_loader):
    # 1.input output
    images = images.cuda(non_blocking=True)
    target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)

    # 2.mixup
    alpha=config.alpha
    lam = np.random.beta(alpha,alpha)
    index = torch.randperm(images.size(0)).cuda()
    inputs = lam*images + (1-lam)*images[index,:]
    targets_a, targets_b = target, target[index]
    outputs = model(inputs)
    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

    # 3.backward
    optimizer.zero_grad()   # reset gradient
    loss.backward()
    optimizer.step()        # update parameters of net
```

我们通过matplotlib来可视化mixup这个过程，两张图片的mixup结果随着lam的变化而发生渐变

```
for i in range(1,10):
    lam= i*0.1
    im_mixup = (im1*lam+im2*(1-lam)).astype(np.uint8)
    plt.subplot(3,3,i)
    plt.imshow(im_mixup)
plt.show()
```

&lt;img src="https://pic3.zhimg.com/v2-4700de06bdcacdcbe4c1460f9fabc4fa_b.jpg" data-caption="" data-size="normal" data-rawwidth="1537" data-rawheight="535" data-default-watermark-src="https://pic2.zhimg.com/v2-157b56b607caf50424ce3a3007ae4589_b.jpg" class="origin_image zh-lightbox-thumb" width="1537" data-original="https://pic3.zhimg.com/v2-4700de06bdcacdcbe4c1460f9fabc4fa_r.jpg"&gt;![img](https://pic3.zhimg.com/80/v2-4700de06bdcacdcbe4c1460f9fabc4fa_hd.jpg)

&lt;img src="https://pic1.zhimg.com/v2-09cdc6d9630128712da6cd571a49045c_b.jpg" data-caption="" data-size="normal" data-rawwidth="1545" data-rawheight="1359" data-default-watermark-src="https://pic4.zhimg.com/v2-c34bf2365135c8239e0395d85be87727_b.jpg" class="origin_image zh-lightbox-thumb" width="1545" data-original="https://pic1.zhimg.com/v2-09cdc6d9630128712da6cd571a49045c_r.jpg"&gt;![img](https://pic1.zhimg.com/80/v2-09cdc6d9630128712da6cd571a49045c_hd.jpg)

实际代码中的lam由随机数生成器控制，lam符合参数为(alpha,alpha)的beta分布，默认取alpha=1，这里的alpha是一个超参数，比如我遇到的一个情况就是alpha=2效果更好，alpha的值越大，生成的lam偏向0.5的可能性更高。

&lt;img src="https://pic2.zhimg.com/v2-5f5f6189ace25ad1432f21030fc3e471_b.jpg" data-caption="" data-size="small" data-rawwidth="1241" data-rawheight="781" data-default-watermark-src="https://pic1.zhimg.com/v2-435eab280f6578678c7adcac22b62138_b.jpg" class="origin_image zh-lightbox-thumb" width="1241" data-original="https://pic2.zhimg.com/v2-5f5f6189ace25ad1432f21030fc3e471_r.jpg"&gt;![img](https://pic2.zhimg.com/80/v2-5f5f6189ace25ad1432f21030fc3e471_hd.jpg)

如上就是图像分类mixup的一个pytorch实现，说完这个我们来看看检测怎么用mixup

------

对于目标检测的话，如果用上面这种图像mixup融合，损失函数加权相加的方式，我想就不存在标签问题了：图1 和 图2 按照比例lam进行线性融合，然后送入model进行检测分别按标签计算损失函数，然后按照lam加权相加得到最终的损失函数。

&lt;img src="https://pic2.zhimg.com/v2-a24e855e639eeb4f3a480ba2b6053789_b.jpg" data-caption="" data-size="normal" data-rawwidth="2293" data-rawheight="1310" data-default-watermark-src="https://pic1.zhimg.com/v2-beaef9796c01e26f4d8af939fd507bac_b.jpg" class="origin_image zh-lightbox-thumb" width="2293" data-original="https://pic2.zhimg.com/v2-a24e855e639eeb4f3a480ba2b6053789_r.jpg"&gt;![img](https://pic2.zhimg.com/80/v2-a24e855e639eeb4f3a480ba2b6053789_hd.jpg)

顺手搜了一下GluonCV版本的中目标检测的mixup实现，我把代码缩减拼接了一下

```python
class MixupDetection(Dataset):
	# mixup two images
	height = max(img1.shape[0], img2.shape[0])
	width = max(img1.shape[1], img2.shape[1])
	mix_img = mx.nd.zeros(shape=(height, width, 3), dtype='float32')
	mix_img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * lambd
	mix_img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1. - lambd)
	mix_img = mix_img.astype('uint8')
	y1 = np.hstack((label1, np.full((label1.shape[0], 1), lambd)))
	y2 = np.hstack((label2, np.full((label2.shape[0], 1), 1. - lambd)))
	mix_label = np.vstack((y1, y2))
	return mix_img, mix_label

from gluoncv.data.mixup import MixupDetection
train_dataset = MixupDetection(train_dataset)

def train(net, train_data, val_data, eval_metric, ctx, args):
        for epoch in range(args.start_epoch, args.epochs):
        mix_ratio = 1.0
        if args.mixup:
            # TODO(zhreshold) only support evenly mixup now, target generator needs to be modified otherwise
            train_data._dataset.set_mixup(np.random.uniform, 0.5, 0.5)
            mix_ratio = 0.5
            if epoch >= args.epochs - args.no_mixup_epochs:
                train_data._dataset.set_mixup(None)
                mix_ratio = 1.0
        for i, batch in enumerate(train_data):
            with autograd.record():
                for data, label, rpn_cls_targets, rpn_box_targets, rpn_box_masks in zip(*batch):
                    # overall losses
                    losses.append(rpn_loss.sum() * mix_ratio + rcnn_loss.sum() * mix_ratio)
                    metric_losses[0].append(rpn_loss1.sum() * mix_ratio)
                    metric_losses[1].append(rpn_loss2.sum() * mix_ratio)
                    metric_losses[2].append(rcnn_loss1.sum() * mix_ratio)
                    metric_losses[3].append(rcnn_loss2.sum() * mix_ratio)
                    add_losses[0].append([[rpn_cls_targets, rpn_cls_targets>=0], [rpn_score]])
                    add_losses[1].append([[rpn_box_targets, rpn_box_masks], [rpn_box]])
                    add_losses[2].append([[cls_targets], [cls_pred]])
                    add_losses[3].append([[box_targets, box_masks], [box_pred]])
                autograd.backward(losses)
train_data, val_data = get_dataloader(
    net, train_dataset, val_dataset, args.batch_size, args.num_workers)
train(net, train_data, val_data, eval_metric, ctx, args)
```

gluoncv中具体的标签处理代码我没怎么看明白，而且代码牵扯面很大，来不及仔细看，但不管怎样，代码里有几个点是明确的：

1. 图片的融合是很明确的逐像素相加，融合得到的新图的尺寸是取两张图片的中的最大值，也就是说(600,800)和(900,700)两张图融合得到的新图大小是(900,800)，新增的部分取零，这一步的意义是确保新图装得下原先的两张图，且不改变检测框的绝对位置；
2. 源代码中的todo注释，表明目标检测的mixup还没更新完。目前代码中所采用的mixup系数是固定的，就是0.5，并没有通过beta分布随机生成系数lam，也就是两张图各占一半权重，从更新日期看这个坑好久没有填了；

相关代码链接：

https://github.com/dmlc/gluon-cv/blob/49be01910a8e8424b017ed3df65c4928fc918c67/gluoncv/data/mixup/detection.py#L65github.com

https://github.com/dmlc/gluon-cv/blob/428ee05d7ae4f2955ef00380a1b324b05e6bc80f/scripts/detection/faster_rcnn/train_faster_rcnn.py#L187github.com