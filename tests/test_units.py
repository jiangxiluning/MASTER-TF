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
# File: test_units.py
# Author: Owen Lu
# Date:
# Email: jiangxiluning@gmail.com
# Description:
from typing import *

import tensorflow as tf
import anyconfig
import easydict
import numpy as np

from src.dataset import utils as data_utils
from src.tools.train_net import get_dataset
from src.model.backbone import Resnet31, BasicBlock
from src.model.model import MasterModel
from src.model.metrics import WordAccuary
from src.dataset.benchmark_data_generator import generator_lmdb

def test_dataset():
    config = anyconfig.load('/home/luning/dev/projects/master-tf/configs/master.yaml')
    config = easydict.EasyDict(config)
    train_ds, eval_ds = get_dataset(config)
    #ds = dataset.LMDBDataset("/home/luning/dev/data/SynthText800k/synth_lmdb", 100, 48)
    for index,v in enumerate(eval_ds):
        print(index)

def test_training():
    pass

def test_backbone():
    config = anyconfig.load('configs/master.yaml')
    config = easydict.EasyDict(config)
    bb_config = config['model']['backbone']

    resnet = Resnet31(block=BasicBlock, backbone_config=bb_config)
    input = tf.random.normal([10, 48, 160, 3])
    output = resnet(input, training=True)
    print(output.shape)

def test_master():
    config = anyconfig.load('configs/master.yaml')
    config = easydict.EasyDict(config)

    image = tf.random.normal([10, 48, 160, 3])
    target = tf.constant(np.random.randint(0,10, (10, 50)), dtype=tf.uint8)
    model = MasterModel(config.model, 10, (48, 160))
    optimizer = tf.optimizers.Adadelta(learning_rate=1.0, rho=0.9, epsilon=1e-6)
    with tf.GradientTape() as tape:
        logits = model(image, target, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(target, logits, from_logits=True)

    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))


    print(logits)

def test_decoder():
    config = anyconfig.load('/home/luning/dev/projects/master-tf/configs/master.yaml')
    config = easydict.EasyDict(config)

    image = tf.random.normal([10, 48, 160, 3])
    model = MasterModel(config.model, 10, (48, 160))

    ys = model.decode(image, padding=tf.constant(True))
    decoded_tensor = data_utils.LabelTransformer.decode_tensor(ys)
    print(decoded_tensor)


def test_accuarcy():
    acc = WordAccuary()

    preds = [['aaa', 'bbb', 'ccc'], ['ddd', 'ccc', 'bbb']]
    gts = [['aaa', 'ccc', 'ddd'], ['aaa', 'ccc', 'bbb']]

    for pred, gt in zip(preds, gts):
        acc.update(pred, gt)

    assert 0.5 == acc.compute()

    acc.reset()

    preds = [['aaa', 'bbb', 'ccc'], ['ddd', 'ccc', 'bbb']]
    gts = [['aaa', 'bbb', 'ccc'], ['ddd', 'ccc', 'bbb']]

    for pred, gt in zip(preds, gts):
        acc.update(pred, gt)

    assert 1.0 == acc.compute()
    acc.reset()

    preds = [['aaa', 'bbb', 'ccc'], ['ddd', 'ccc', 'bbb']]
    gts = [['aad', 'bbd', 'cc'], ['dd', 'cc', 'bb']]
    for pred, gt in zip(preds, gts):
        acc.update(pred, gt)
    assert 0.0 == acc.compute()

def test_benchmark_dataset():
    for i in generator_lmdb('/data/ocr/reg/evaluation/IC15_2077', rgb=False):
        print(i)

def test_hashtable():
    keys_tensor = tf.constant(list(data_utils.LabelTransformer.dict.values()))
    vals_tensor = tf.constant(list(data_utils.LabelTransformer.dict.keys()))
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor, key_dtype=tf.int32, value_dtype=tf.string),
        default_value=tf.constant('<UNK>')
    )

    inputs = tf.random.uniform(shape=[3,3,3], minval=0, maxval=len(data_utils.LabelTransformer.dict.keys())-1, dtype=tf.int32)
    print(table.lookup(inputs))