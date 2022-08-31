#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 user <user@4029GP-TR>
#
# Distributed under terms of the MIT license.

import copy
from ..builder import PIPELINES
import albumentations
from albumentations import Compose
import mmcv


@PIPELINES.register_module()
class Albu(object):
    def __init__(self, transforms, keymap=None, update_pad_shape=False):
        if Compose is None:
            raise RuntimeError('albumentations is not installed')
        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)
        if keymap is not None:
            keymap = copy.deepcopy(keymap)
        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.aug = Compose([self.albu_builder(t) for t in self.transforms])
        if not keymap:
            self.keymap_to_albu = {'img': 'image', 'gt_semantic_seg': 'mask'}
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}
    def albu_builder(self, cfg):
        """Import a module from albumentations.
        It inherits some of :func:`build_from_cfg` logic.
        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """
        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()
        obj_type = args.pop('type')
        if mmcv.is_str(obj_type):
            obj_cls = getattr(albumentations, obj_type)
        else:
            raise TypeError(f'type must be str, but got {type(obj_type)}')
        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]
        return obj_cls(**args)
    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper.
        Renames keys according to keymap provided.
        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """
        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict
    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)
        results = self.aug(**results)
        # back to the original format
        results = self.mapper(results, self.keymap_back)
        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape
        return results
    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str

