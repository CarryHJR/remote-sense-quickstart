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
label_paths = glob.glob('/home/user/workplace/python/mmsegmentation-0.19.0/work_dirs/rsipac2022_stage1_baseline/convnext_l_bs32_40k_ms/results/*.png')
def process(label_path):
    label = np.array(Image.open(label_path), dtype=np.int32)
    label = label * 100
    label = Image.fromarray(label)
    label.save(label_path)
_ = mmcv.track_parallel_progress(process, label_paths, 8)
