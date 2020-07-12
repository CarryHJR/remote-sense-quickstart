## README
数据集是采用2019年遥感图像稀疏表征与智能分析竞赛中的变化检测数据集，原数据有四个通道，本文只是做一个demo，所以一切从简，代码放在了 [https://github.com/CarryHJR/remote-sense-quickstart](https://github.com/CarryHJR/remote-sense-quickstart)
感谢star

## 数据集可视化
```
from libtiff import TIFF
import matplotlib.pyplot as plt
from skimage.transform import match_histograms

import numpy as np
def normalize(x):
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

root = '/home/yons/data/change-detection/rssrai2019_change_detection/train/train'
import glob
from libtiff import TIFF
from skimage.transform import match_histograms

image_path_list_2018 = glob.glob(root+'/img_2018'+'/*.tif')
image_path_list_2018.sort()
image_path_list_2017 = glob.glob(root+'/img_2017'+'/*.tif')
image_path_list_2017.sort()
image_path_list_mask = glob.glob(root+'/mask'+'/*.tif')
image_path_list_mask.sort()

for idx in range(18):
    
    image1 = TIFF.open(image_path_list_2017[idx], mode='r').read_image()[:, :, [2, 1, 0]]
    image2 = TIFF.open(image_path_list_2018[idx], mode='r').read_image()[:, :, [2, 1, 0]]
    image2 = match_histograms(image2, image1, multichannel=True)
    label = TIFF.open(image_path_list_mask[idx], mode='r').read_image()
    label = (label / 255).astype(np.uint8)
    
    fig, axes = plt.subplots(1,3, figsize=(15,5))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
    axes[0].imshow(normalize(image1))
    axes[0].axis('off')
    axes[1].imshow(normalize(image2))
    axes[1].axis('off')
    axes[2].imshow(label)
    axes[2].axis('off')
 
plt.show()
```
效果如下:
![从简显示第一个图](https://upload-images.jianshu.io/upload_images/141140-02e559e22adf7aaa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 定义网络结构
as we know, 一些论文里用siamese网络，个人经过试验感觉效果不理想，还是暴力的unet好用

```
import torch
from torch import nn


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = nn.functional.interpolate(x1, scale_factor=2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = double_conv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), double_conv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), double_conv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), double_conv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), double_conv(512, 512))

        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
from torchsummary import summary
net = UNet(6,2)
summary(net, input_size=[(3, 960, 960), (3, 960, 960)], batch_size=1, device='cpu')
```
## 定义dataset
```
import itertools
import glob
from libtiff import TIFF
import numpy as np
from torch.utils.data import Dataset

from skimage.transform import match_histograms
import torch
class SctDataset(Dataset):
    def __init__(self, root, size=512, slide=256):
        image_path_list_2018 = glob.glob(root + '/img_2018' + '/*.tif')
        image_path_list_2018.sort()
        image_path_list_2017 = glob.glob(root + '/img_2017' + '/*.tif')
        image_path_list_2017.sort()
        image_path_list_mask = glob.glob(root + '/mask' + '/*.tif')
        image_path_list_mask.sort()
        self.image_path_list_2018 = image_path_list_2018
        self.image_path_list_2017 = image_path_list_2017
        self.image_path_list_mask = image_path_list_mask

    def __getitem__(self, idx):

        image1 = TIFF.open(self.image_path_list_2017[idx], mode='r').read_image()[:, :, [2, 1, 0]].transpose((2, 0, 1))
        image2 = TIFF.open(self.image_path_list_2018[idx], mode='r').read_image()[:, :, [2, 1, 0]].transpose((2, 0, 1))
        image2 = match_histograms(image2, image1, multichannel=True)
        label = TIFF.open(self.image_path_list_mask[idx], mode='r').read_image()
        label = (label / 255).astype(np.uint8)

        return torch.from_numpy(image1.astype(np.float32)), torch.from_numpy(image2.astype(np.float32)), torch.from_numpy(label).long()

    def __len__(self):
        return len(self.image_path_list_2018)
from torch.utils.data import DataLoader
test_loader = DataLoader(SctDataset('/home/yons/data/change-detection/rssrai2019_change_detection/train/train'), batch_size=1, shuffle=True, num_workers=1)
test_iter = iter(test_loader)
x1, x2, target = next(test_iter)
print(x1.shape,x2.shape,target.shape)
```
输出是
```
torch.Size([1, 3, 960, 960]) torch.Size([1, 3, 960, 960]) torch.Size([1, 960, 960])
```
和预想中一样，let's move on.

## 定义TverskyLoss
最开始用的criterion = nn.CrossEntropyLoss()，误判很严重
然后用的TverskyLoss(alpha=0.1, beta=0.9) 发现偏向于change，最终采用0.3 0.7
```
import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, eps=1e-7, size_average=True):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.size_average = size_average
        self.eps = eps

    def forward(self, logits, true):
        """Computes the Tversky loss [1].
        Args:
            true: a tensor of shape [B, H, W] or [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            alpha: controls the penalty for false positives.
            beta: controls the penalty for false negatives.
            eps: added to the denominator for numerical stability.
        Returns:
            tversky_loss: the Tversky loss.
        Notes:
            alpha = beta = 0.5 => dice coeff
            alpha = beta = 1 => tanimoto coeff
            alpha + beta = 1 => F beta coeff
        References:
            [1]: https://arxiv.org/abs/1706.05721
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)

        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        fps = torch.sum(probas * (1 - true_1_hot), dims)
        fns = torch.sum((1 - probas) * true_1_hot, dims)
        num = intersection
        denom = intersection + (self.alpha * fps) + (self.beta * fns)
        tversky_loss = (num / (denom + self.eps)).mean()
        return (1 - tversky_loss)
criterion = TverskyLoss(alpha=0.3, beta=0.7)
y = net(x1,x2)
loss = criterion(y, target)
print(loss.item())
```
## 跑跑训练
```
from tqdm import tqdm_notebook as tqdm
import torch
train_loader = DataLoader(SctDataset('/home/yons/data/change-detection/rssrai2019_change_detection/train/train'), batch_size=1, shuffle=True, num_workers=1)
val_loader = DataLoader(SctDataset('/home/yons/data/change-detection/rssrai2019_change_detection/val/val'), batch_size=1, shuffle=True, num_workers=1)
print('train: ', len(train_loader))
print('val: ', len(val_loader))

best_loss = 9999

save_dir = '/home/yons/workplace/python/change-detection/log/0.4'
import shutil, os
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    shutil.rmtree(save_dir)
    os.makedirs(save_dir)
        
net = UNet(6, 2)
net = net.cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 8, 15], gamma=0.1)

for epoch in range(20):
    net.train()
    train_loss = 0
    for step, data in tqdm(enumerate(train_loader)):
        x1, x2, target = data
        x1, x2, target = x1.cuda(), x2.cuda(), target.cuda()
        optimizer.zero_grad()
        y = net(x1, x2)
        loss = criterion(y, target)
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()
    train_loss = train_loss / len(train_loader)
    net.eval()
    with torch.no_grad():
        val_loss = 0
        for step, data in tqdm(enumerate(val_loader)):
            x1, x2, target = data
            x1, x2, target = x1.cuda(), x2.cuda(), target.cuda()
            y = net(x1, x2)
            loss = criterion(y, target)
            val_loss = val_loss + loss.item()
        val_loss = val_loss / len(val_loader)
    print(epoch, train_loss, val_loss)
    if val_loss < best_loss:
        best_loss = val_loss
        save_path = os.path.join(save_dir, 'best_loss.pth')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'loss': best_loss
        }
        torch.save(state, save_path)
    scheduler.step()
```
最终loss是
```
train: 0.45967846115430194  | val: 0.39102962613105774
```
## 加载保存好的模型
```
net = UNet(6, 2)
net = net.cuda()
checkpoint = torch.load('/home/yons/workplace/python/change-detection/log/0.4/best_loss.pth')
net.load_state_dict(checkpoint['net'])
print(checkpoint['loss'], checkpoint['epoch'])
```
## 预测及可视化
```py
def normalize(x):
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
net.eval()
with torch.no_grad():
    val_loss = 0
    for step, data in tqdm(enumerate(val_loader)):
        x1, x2, target = data
        x1, x2, target = x1.cuda(), x2.cuda(), target.cuda()
        y = net(x1, x2)
        loss = criterion(y, target)
        val_loss = val_loss + loss.item()
        
        x1 = x1.cpu().numpy()[0].transpose((1,2,0))
        x2 = x2.cpu().numpy()[0].transpose((1,2,0))
        _, y = torch.max(y, 1)
        y = y.cpu().numpy()[0]
        target = target.cpu().numpy()[0]
        
        fig, axes = plt.subplots(1,4, figsize=(8,2))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.1)
        
        image1 = TIFF.open(image_path_list_2017[step], mode='r').read_image()[:,:,[2,1,0]]
        image2 = TIFF.open(image_path_list_2018[step], mode='r').read_image()[:,:,[2,1,0]]
        image2 = match_histograms(image2, image1, multichannel=True)
        axes[0].imshow(normalize(image1))
        axes[0].axis('off')
        axes[1].imshow(normalize(image2))
        axes[1].axis('off')
        axes[2].imshow(target)
        axes[2].axis('off')
        axes[3].imshow(y)
        axes[3].axis('off')
        
        plt.show()
    val_loss = val_loss / len(val_loader)
print(val_loss)
```
![image.png](https://upload-images.jianshu.io/upload_images/141140-7f2443c29cc81e9d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
看起来还行，发现可视化的图片路径错了，mask没有问题，不改了，大家参考的时候注意一下.
