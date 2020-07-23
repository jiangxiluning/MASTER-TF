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
# File: transformer.py
# Author: Owen Lu
# Email: jiangxiluning@gmail.com
# Description:
from typing import *
import copy

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def clones(module, N):
    "Produce N identical layers."
    return [copy.deepcopy(module) for _ in range(N)]


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

@tf.function
def create_padding_mask(seq, padding=0):
    seq = tf.cast(tf.math.equal(seq, padding), tf.float32)

    # 添加额外的维度来将填充加到
    # 注意力对数（logits）。
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

@tf.function
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, h, d_model, dropout=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(name='MultiHeadAttention', **kwargs)
        assert d_model % h == 0

        self.d_k = int(d_model / h)
        self.h = h
        self.linears = clones(keras.layers.Dense(d_model), 4)
        self.attn = None
        self.dropout = dropout

    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, mask: tf.Tensor, training=False):
        B = tf.shape(query)[0]

        qkv = []
        for l, x in zip(self.linears, (query, key, value)):
            x = tf.reshape(x, shape=(B, -1, self.h, self.d_k))
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            qkv.append(x)


        # attention
        query = qkv[0]
        key = qkv[1]
        value = qkv[2]

        d_k = value.shape[-1]
        score = tf.matmul(query, tf.transpose(key, perm=(0, 1, 3, 2))) / tf.sqrt(tf.cast(d_k, dtype=query.dtype))

        if mask is not None:
            score += mask * -1e9

        p_attn = keras.activations.softmax(score, axis=-1)
        p_attn = keras.layers.Dropout(rate=self.dropout)(p_attn, training=training)
        x = tf.matmul(p_attn, value)

        self.attn = p_attn


        x = tf.transpose(x, perm=(0, 2, 1, 3))
        x = tf.reshape(x, shape=(B, -1, self.h * self.d_k))

        return self.linears[-1](x)

class PositionwiseFeedForward(keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """

        Args:
            d_model:
            d_ff:
            dropout:
        """
        super(PositionwiseFeedForward, self).__init__(name='PositionwiseFeedForward')
        self.dense_1 = keras.layers.Dense(d_ff)
        self.dense_2 = keras.layers.Dense(d_model)
        self.dropout = keras.layers.Dropout(rate=dropout)

    def call(self, inputs, training):
        return self.dense_2(self.dropout(keras.activations.relu(self.dense_1(inputs)), training=training))

class SublayerConnection(keras.layers.Layer):
    def __init__(self, dropout):
        super(SublayerConnection, self).__init__(name='SublayerConnection')
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, x, sublayer, training=False):
        return x + self.dropout(sublayer(self.norm(x), training), training=training)


class DecoderLayer(keras.layers.Layer):
    def __init__(self, self_attn, src_attn, feed_forard, dropout, **kwargs):
        super().__init__(name='DecoderLayer', **kwargs)
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forard
        self.sublayer = clones(SublayerConnection(dropout), 3)

    def call(self, x, memory, src_mask, tgt_mask, training=False):
        x = self.sublayer[0](x, lambda x, training: self.self_attn(x, x, x, tgt_mask, training=training), training=training)
        x = self.sublayer[1](x, lambda x, training: self.src_attn(x, memory, memory, src_mask, training=training), training=training)
        return self.sublayer[2](x, self.feed_forward)

class Decoder(keras.layers.Layer):
    def __init__(self, d_model:int, headers:int, position_num:int, config):
        super(Decoder, self).__init__(name='Decoder')

        self.N = config.stacks
        self.dropout_rate = config.dropout
        self.d_ff = config.feed_forward_size

        self.src_attn = MultiHeadAttention(headers, d_model, dropout=self.dropout_rate)
        self.self_attn = MultiHeadAttention(headers, d_model, dropout=self.dropout_rate)
        self.ff = PositionwiseFeedForward(d_model, self.d_ff, dropout=self.dropout_rate)
        self.decoder_layer = DecoderLayer(
                                          src_attn=self.src_attn,
                                          self_attn=self.self_attn,
                                          feed_forard=self.ff,
                                          dropout=self.dropout_rate)

        self.decoder_pe = positional_encoding(position_num, d_model)

        self.layers = clones(self.decoder_layer, self.N)
        self.norm = keras.layers.LayerNormalization()

    def call(self, x, memory, src_mask, tgt_mask, training=False):
        T = tf.shape(x)[1]
        x = x + self.decoder_pe[:, :T]

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, training=training)
        return self.norm(x)