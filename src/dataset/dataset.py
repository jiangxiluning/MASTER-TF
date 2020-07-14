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
# File: dataset.py
# Author: Owen Lu
# Date:
# Email: jiangxiluning@gmail.com
# Description:
from typing import *
import pathlib
import functools
import math

import tensorflow as tf
import lmdb

from . import lmdb_data_generator
from .utils import LabelTransformer, resize_width

class LmdbDataset:

    def __init__(self,
                 lmdb_generator: Callable,
                 lmdb_paths: Union[Dict[str, str], str],
                 rgb: bool,
                 image_width: int,
                 image_height: int,
                 batch_size: int,
                 workers: int,
                 drop_last=True):
        self.num_samples = 0

        if isinstance(lmdb_paths, dict):
            datasets = []
            for name, path in lmdb_paths.items():
                lmdb_path = pathlib.Path(path)
                if not lmdb_path.exists():
                    raise FileNotFoundError('LMDB file is not found. {}'.format(lmdb_path))
                env = lmdb.open(lmdb_path.as_posix(), max_readers=32, readonly=True)
                with env.begin() as txn:
                    try:
                        num_samples = int(txn.get(b"nSamples").decode())
                    except:
                        num_samples = int(txn.get(b'num-samples').decode())
                self.num_samples += num_samples

                ds = tf.data.Dataset.from_generator(generator=lmdb_generator,
                                                    output_types=(tf.float32, tf.int32),
                                                    output_shapes=(tf.TensorShape([None, None, 1]),
                                                                   tf.TensorShape([LabelTransformer.max_length, ])),
                                                    args=(lmdb_path.as_posix(), rgb))
                datasets.append(ds)

            self.ds = functools.reduce(lambda x,y: x.concatenate(y), datasets)
        elif isinstance(lmdb_paths, str):
            lmdb_path = pathlib.Path(lmdb_paths)
            if not lmdb_path.exists():
                raise FileNotFoundError('LMDB file is not found. {}'.format(lmdb_path))
            env = lmdb.open(lmdb_path.as_posix(), max_readers=32, readonly=True)
            with env.begin() as txn:
                try:
                    num_samples = int(txn.get(b"nSamples").decode())
                except:
                    num_samples = int(txn.get(b'num-samples').decode())
            self.num_samples += num_samples

            self.ds = tf.data.Dataset.from_generator(generator=lmdb_generator,
                                                     output_types=(tf.float32, tf.int32),
                                                     output_shapes=(tf.TensorShape([None, None, 1]),
                                                                    tf.TensorShape([LabelTransformer.max_length, ])),
                                                     args=(lmdb_path.as_posix(), rgb))

        strategy = tf.distribute.get_strategy()
        batch_size = batch_size * strategy.num_replicas_in_sync
        self.steps = math.floor(self.num_samples/batch_size) if drop_last else math.ceil(self.num_samples/batch_size)

        training_transform = functools.partial(resize_width,
                                               width=image_width,
                                               height=image_height)
        def f(x,y):
            return x,y

        self.ds = self.ds.map(map_func=training_transform,
                              num_parallel_calls=workers) \
                         .apply(tf.data.experimental.ignore_errors()) \
                         .shuffle(buffer_size=workers * batch_size) \
                         .batch(batch_size=batch_size, drop_remainder=drop_last) \
                         .map(map_func=f, num_parallel_calls=workers)


        self.ds = strategy.experimental_distribute_dataset(self.ds)

    def __iter__(self):
        return iter(self.ds)


