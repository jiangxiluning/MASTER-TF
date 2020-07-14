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
# File: utils.py
# Author: Owen Lu
# Date: 
# Email: jiangxiluning@gmail.com
# Description:
from typing import *

import tensorflow as tf

@tf.function
def greedy_decoding(logits: tf.Tensor,
                    memory: tf.Tensor,
                    max_len: tf.int32,
                    start_symbol: tf.int32,
                    padding_symbol: tf.int32,
                    padding: tf.bool=tf.constant(False, dtype=tf.bool)):
    B = tf.shape(logits)[0]

    if padding:
        tf.Assert(padding_symbol, data=padding_symbol)
        tf.Assert(max_len>0, data=max_len)

        ys = tf.fill(dims=(B, max_len + 2), value=padding_symbol)
        ys[:, 0] = start_symbol
    else:
        if tf.equal(max_len, tf.constant(-1)):
            max_len = tf.constant(100, dtype=tf.int32)
        ys = tf.fill(dims=(B, 1), value=start_symbol)

    for i in range(max_len + 1):
        pass



