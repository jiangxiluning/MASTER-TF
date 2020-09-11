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
# File: gcb.py
# Author: Owen Lu
# Date:
# Email: jiangxiluning@gmail.com
# Description:
from typing import *

import tensorflow as tf


class GolbalContextBlock(tf.keras.layers.Layer):

    def __init__(self,
                 inplanes,
                 ratio,
                 headers,
                 pooling_type='att',
                 att_scale=False,
                 fusion_type='channel_add',
                 **kwargs):
        """

        Args:
            inplanes:
            ratio:
            headers:
            pooling_type:
            att_scale:
            fusion_type:
            **kwargs:
        """

        super().__init__(name='GCB', **kwargs)
        assert pooling_type in ['att', 'avg']
        assert fusion_type in ['channel_add', 'channel_concat', 'channel_mul']
        assert inplanes % headers == 0 and inplanes >= 8

        self.headers = headers
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_type
        self.att_scale = att_scale

        self.single_header_inplanes = int(inplanes / headers)

        if self.pooling_type == 'att':
            self.conv_mask = tf.keras.layers.Conv2D(1,
                                                    kernel_size=1,
                                                    kernel_initializer=tf.initializers.he_normal())
        else:
            self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=1)

        if self.fusion_type == 'channel_add':
            self.channel_add_conv = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(self.planes, kernel_size=1,
                                           kernel_initializer=tf.initializers.he_normal()),
                    tf.keras.layers.LayerNormalization([1, 2, 3]),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2D(self.inplanes, kernel_size=1,
                                           kernel_initializer=tf.initializers.he_normal()),
                ],
                name='channel_add_conv'
            )
        elif self.fusion_type == 'channel_concat':
            self.channel_concat_conv = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(self.planes,
                                           kernel_size=1,
                                           kernel_initializer=tf.initializers.he_normal()),
                    tf.keras.layers.LayerNormalization([1,2,3]),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2D(self.inplanes,
                                           kernel_size=1,
                                           kernel_initializer=tf.initializers.he_normal()),
                ],
                name='channel_concat_conv'
            )
            self.cat_conv = tf.keras.layers.Conv2D(self.inplanes,
                                                   kernel_size=1,
                                                   kernel_initializer=tf.initializers.he_normal())
            self.layer_norm = tf.keras.layers.LayerNormalization(axis=[1,2,3])
        else:
            self.channel_mul_conv = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(self.planes,
                                           kernel_size=1,
                                           kernel_initializer=tf.initializers.he_normal()),
                    tf.keras.layers.LayerNormalization([1, 2, 3]),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2D(self.inplanes,
                                           kernel_size=1,
                                           kernel_initializer=tf.initializers.he_normal()),
                ],
                name='channel_mul_conv'
            )

    @tf.function
    def spatial_pool(self, inputs: tf.Tensor):
        B = tf.shape(inputs)[0]
        H = tf.shape(inputs)[1]
        W = tf.shape(inputs)[2]
        C = tf.shape(inputs)[3]

        if self.pooling_type == 'att':

            # B, H, W, h, C/h
            x = tf.reshape(inputs, shape=(B, H, W, self.headers, self.single_header_inplanes))
            # B, h, H, W, C/h
            x = tf.transpose(x, perm=(0, 3, 1, 2, 4))

            # B*h, H, W, C/h
            x = tf.reshape(x, shape=(B*self.headers, H, W, self.single_header_inplanes))

            input_x = x

            # B*h, 1, H*W, C/h
            input_x = tf.reshape(input_x, shape=(B*self.headers, 1, H*W, self.single_header_inplanes))
            # B*h, 1, C/h, H*W
            input_x = tf.transpose(input_x, perm=[0, 1, 3, 2])

            # B*h, H, W, 1,
            context_mask = self.conv_mask(x)

            # B*h, 1, H*W, 1
            context_mask = tf.reshape(context_mask, shape=(B*self.headers, 1, H*W, 1))

            # scale variance
            if self.att_scale and self.headers > 1:
                context_mask = context_mask / tf.sqrt(self.single_header_inplanes)

            # B*h, 1, H*W, 1
            context_mask = tf.keras.activations.softmax(context_mask, axis=2)

            # B*h, 1, C/h, 1
            context = tf.matmul(input_x, context_mask)
            context = tf.reshape(context, shape=(B, 1, C, 1))

            # B, 1, 1, C
            context = tf.transpose(context, perm=(0, 1, 3, 2))
        else:
            # B, 1, 1, C
            context = self.avg_pool(inputs)

        return context


    def call(self, inputs, **kwargs):
        # B, 1, 1, C
        context = self.spatial_pool(inputs)

        out = inputs
        if self.fusion_type == 'channel_mul':
            channel_mul_term = tf.sigmoid(self.channel_mul_conv(context))
            out = channel_mul_term * out
        elif self.fusion_type == 'channel_add':
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        else:
            # B, 1, 1, C
            channel_concat_term = self.channel_concat_conv(context)


            B = tf.shape(out)[0]
            H = tf.shape(out)[1]
            W = tf.shape(out)[2]
            C = tf.shape(out)[3]
            out = tf.concat([out, tf.broadcast_to(channel_concat_term, shape=(B, H, W, C))], axis=-1)
            out = self.cat_conv(out)
            out = self.layer_norm(out)
            out = tf.keras.activations.relu(out)

        return out


