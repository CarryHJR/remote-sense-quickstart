# 赛题简介
http://rsipac.whu.edu.cn/subject_three 武大测绘遥感国重实验室举办的 遥感语义分割比赛

# 环境
```
git clone https://github.com/open-mmlab/mmsegmentation.git
git checkout tags/v0.19.0 -b v0.19.0
pip install -e .
```

8卡3090 

# 执行


0. `python preprocess.py` 处理一下标签
1. 将本目录下的`mmseg/datasets/custom.py`替换`mmsegmentation/mmseg/datasets/custom.py` ; `mmseg/models/backbones/convnext.py` 和 `mmseg/models/backbones/__init__.py` 复制到 `mmsegmentation/mmseg/models/backbones/`
2. 在本目录下，多卡 执行 `./tools/dist_traintest.sh configs/rsipac2022_stage1_baseline/convnext_l_bs32_40k_ms.py 8`
3. `python proprocess.py` 转换一下标签
4. 进入`work_dirs/rsipac2022_stage1_baseline/convnext_l_bs32_40k_ms`文件夹，执行`7z a results.zip results`，然后将results.zip提交就是线上71.8左右


# 主要改动
train.py
* 将默认的validate改为了no-validate，即不加`--validate`就不会在train过程中evaluate
* 加入了统计时间，写到了log里面
* 将work_dir的位置自动设置为和config同级别的目录

test.py
* 将format-only设置了默认
* 将生成的预测结果自动放到了work_dir目录下


traintest.sh
* 整合了 train 和 test

mmseg/datasets/custom.py
* 加入了format_result功能

mmseg/models/backbones/convnext.py
* 加入convnext

# 核心trick
* convnext large 22k
* aug
* 多尺度训练测试
