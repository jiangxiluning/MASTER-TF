#   ______                                           __                 
#  /      \                                         /  |                
# /$$$$$$  | __   __   __   ______   _______        $$ |       __    __ 
# $$ |  $$ |/  | /  | /  | /      \ /       \       $$ |      /  |  /  |
# $$ |  $$ |$$ | $$ | $$ |/$$$$$$  |$$$$$$$  |      $$ |      $$ |  $$ |
# $$ |  $$ |$$ | $$ | $$ |$$    $$ |$$ |  $$ |      $$ |      $$ |  $$ |
# $$ \__$$ |$$ \_$$ \_$$ |$$$$$$$$/ $$ |  $$ |      $$ |_____ $$ \__$$ |
# $$    $$/ $$   $$   $$/ $$       |$$ |  $$ |      $$       |$$    $$/ 
#  $$$$$$/   $$$$$/$$$$/   $$$$$$$/ $$/   $$/       $$$$$$$$/  $$$$$$/ 
#
# File: eval_net.py
# Author: Owen Lu
# Date: 
# Email: jiangxiluning@gmail.com
# Description:
from typing import *

import click
import tensorflow as tf
import numpy as np
from easydict import EasyDict
import anyconfig

from ..model.model import MasterModel
from ..dataset import utils as dataset_utils

class Predictor:

    def __init__(self, ckpt:str, config:dict, img_width:int, img_height:int):
        self.img_width = img_width
        self.img_height = img_height

        self.model = MasterModel(config.model,
                            dataset_utils.LabelTransformer.nclass,
                            (config.dataset.train.width, config.dataset.train.height))

        checkpoint = tf.train.Checkpoint(model=self.model)
        status = checkpoint.restore(ckpt)
        status.expect_partial()

    def predict(self, image:tf.Tensor):
        image, _ = dataset_utils.resize_width(image, None, width=self.img_width, height=self.img_height)

        pred, _ = self.model.decode(image)
        pred: List = dataset_utils.LabelTransformer.decode_tensor(pred.numpy())
        return pred


    @classmethod
    def checkpoint_to_saved_model(cls, ckpt_dir:str, config_file:str, output_path:str):
        cfg = anyconfig.load(config_file)
        config = EasyDict(cfg)

        model = MasterModel(config.model,
                            dataset_utils.LabelTransformer.nclass,
                            (config.dataset.train.width, config.dataset.train.height))

        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore(ckpt_dir)
        status.expect_partial()


        model([tf.random.normal([1, 48, 160, 1], dtype=tf.float32, name='image'), tf.constant(np.ones([1, 52]), dtype=tf.int32, name='transcript')])
        signatures = {
            'decode': model.decode.get_concrete_function(tf.TensorSpec([None, 48, 160, 1], dtype=tf.float32, name='image')
                                                         )
        }
        tf.saved_model.save(model, output_path, signatures=signatures)
        print('Model is saved at {}'.format(output_path))
