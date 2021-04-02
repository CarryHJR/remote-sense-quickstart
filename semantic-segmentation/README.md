# 复赛docker本地测试方法
先将server.py里面的端口改一下
```
my_infer.run(debuge=False, ip="0.0.0.0")
```
然后docker运行的时候加上`-p 8080:8080`

然后以 `suichang_round1_test_partA_210120/003000.tif` 为例，用自己的模型推理方式推理一遍，算一下`np.sum(label)`

然后运行这段
```python
import base64
import requests
from io import BytesIO
from PIL import Image
import json


def loader2json(data):
    send_json = {}
    bast64_data = base64.b64encode(data)
    bast64_str = str(bast64_data,'utf-8')
    send_json['img'] = bast64_str
    send_json = json.dumps(send_json)
    return send_json

img_path = 'suichang_round1_test_partA_210120/003000.tif'
fin=open(img_path,'rb')
img=fin.read()
data_json = loader2json(img)
url = "http://127.0.0.1:8080/tccapi"
res = requests.post(url, data_json, timeout=3)
bast64_data = res.text.encode(encoding='utf-8')
bytesIO = BytesIO()
label = np.array(Image.open(BytesIO(bytearray(base64.b64decode(bast64_data)))))
print(np.sum(label))

```
看一下两次的sum是否一致即可

# environment
mmseg v0.10.0

# 比赛链接
https://tianchi.aliyun.com/competition/entrance/531860/rankingList

# 线上分数
0.37

# 实验环境
2*2080ti 8h

# 如何复现
```
git clone https://github.com/open-mmlab/mmsegmentation.git
git checkout tags/v0.10.0 -b v0.10.0
pip install -e .
```
1. 将本目录下的`mmseg/datasets/custom.py`替换`mmsegmentation/mmseg/datasets/custom.py`
2. 官网的数据集下载解压后 改动本目录下`configs/TianchiSeg/baseline.py` 84、85、98行的数据集位置即可，不需要对数据集预处理
3. 在目录下，单卡执行`python tool/train.py configs/TianchiSeg/baseline.py`，两卡执行`./tools/dist_train.sh configs/TianchiSeg/baseline.py 2`
4. 训练完后，执行`python tools/test.py configs/TianchiSeg/baseline.py work_dirs/TianchiSeg/baseline/latest.pth`, 在`work_dirs/TianchiSeg/baseline/results`里面就是生成的预测图，进入`results`文件夹，执行`7z a ../results.zip .`，然后将results.zip提交就是线上0.37

# 主要改动
```
├── configs
│   └── TianchiSeg
│       └── baseline.py
├── mmseg
│   └── datasets
│       └── custom.py
├── tools
│   ├── dist_test.sh
│   ├── dist_train.sh
│   ├── test.py
│   └── train.py
└── work_dirs
    └── TianchiSeg
```
train.py
* 将默认的validate改为了no-validate，即不加`--validate`就不会在train过程中evaluate
* 加入了统计时间，写到了log里面
* 将work_dir的位置自动设置为和config同级别的目录

test.py
* 将format-only设置了默认
* 将生成的预测结果放到了work_dir目录下

mmseg/datasets/custom.py
* 加入了format_result功能

# 核心文件
`configs/TianchiSeg/baseline.py`

deeplabv3+ r101 水平、竖直翻转, bs 16 80000 iter

官网的数据集下载解压后 改动84、85、98行的数据集位置即可，不需要对数据集预处理


