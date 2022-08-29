#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 user <user@4029GP-TR>
#
# Distributed under terms of the MIT license.


import glob
from PIL import Image
import numpy as np
import mmcv

import glob
label_paths = glob.glob('/home/user/data/rsipac2022/guangwangxiazai/seg/train/labels/*.png')
def process(label_path):
    label = np.array(Image.open(label_path))
    label = label // 100
    label = Image.fromarray(label)
    label.save(label_path.replace('labels', 'labels_9'))
_ = mmcv.track_parallel_progress(process, label_paths, 8)
