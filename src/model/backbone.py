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
# File: backbone.py
# Author: Owen Lu
# Date:
# Email: jiangxiluning@gmail.com
# Description:
from typing import *

from tensorflow import keras
import tensorflow as tf

from .gcb import GolbalContextBlock

def conv33(out_planes, stride=1):
    return keras.layers.Conv2D(out_planes, kernel_size=3, strides=stride, padding='same', use_bias=False)

class BasicBlock(keras.layers.Layer):
    expansion = 1

    def __init__(self,
                 planes,
                 stride=1,
                 downsample=None,
                 gcb_config=None,
                 use_gcb=None,  **kwargs):
        super().__init__(name='BasciBlock', **kwargs)
        self.conv1 = conv33(planes, stride)
        self.bn1 = keras.layers.BatchNormalization(momentum=0.1,
                                                   epsilon=1e-5)
        self.relu = keras.layers.ReLU()
        self.conv2 = conv33(planes, stride)
        self.bn2 = keras.layers.BatchNormalization(momentum=0.1,
                                                   epsilon=1e-5)
        if downsample:
            self.downsample = downsample
        else:
            self.downsample = tf.identity

        self.stride = stride

        if use_gcb:
            self.gcb = GolbalContextBlock(
                inplanes=planes,
                ratio=gcb_config['ratio'],
                headers=gcb_config['headers'],
                pooling_type=gcb_config['pooling_type'],
                fusion_type=gcb_config['fusion_type'],
                att_scale=gcb_config['att_scale']
            )
        else:
            self.gcb = tf.identity

    def call(self, inputs, **kwargs):

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        out = self.gcb(out)
        out = out + self.downsample(inputs)

        out = self.relu(out)
        return out


class Resnet31(keras.layers.Layer):
    def __init__(self, block, backbone_config, **kwargs):
        super(Resnet31, self).__init__(name='ResNet31', **kwargs)
        layers = [1, 2, 5, 3]
        gcb_config = backbone_config['gcb']
        gcb_enabling = gcb_config['layers']

        self.inplanes = 128
        self.conv1 = keras.layers.Conv2D(64,
                                         kernel_size=3,
                                         padding='same',
                                         use_bias=False,
                                         kernel_initializer=keras.initializers.he_normal())
        self.bn1 = keras.layers.BatchNormalization(momentum=0.9,
                                                   epsilon=1e-5)
        self.relu1 = keras.layers.ReLU()


        self.conv2 = keras.layers.Conv2D(128,
                                         kernel_size=3,
                                         padding='same',
                                         use_bias=False,
                                         kernel_initializer=keras.initializers.he_normal())
        self.bn2 = keras.layers.BatchNormalization(momentum=0.9,
                                                   epsilon=1e-5)
        self.relu2 = keras.layers.ReLU()
        self.maxpool1 = keras.layers.MaxPool2D(strides=2)
        self.layer1 = self._make_layer(block,
                                       256,
                                       layers[0],
                                       stride=1,
                                       use_gcb=gcb_enabling[0],
                                       gcb_config=gcb_config)

        self.conv3 = keras.layers.Conv2D(256,
                                         kernel_size=3,
                                         padding='same',
                                         use_bias=False,
                                         kernel_initializer=keras.initializers.he_normal())
        self.bn3 = keras.layers.BatchNormalization(momentum=0.9,
                                                   epsilon=1e-5)
        self.relu3 = keras.layers.ReLU()
        self.maxpool2 = keras.layers.MaxPool2D(strides=2)
        self.layer2 = self._make_layer(block,
                                       256,
                                       layers[1],
                                       stride=1,
                                       use_gcb=gcb_enabling[1],
                                       gcb_config=gcb_config)


        self.conv4 = keras.layers.Conv2D(256,
                                         kernel_size=3,
                                         padding='same',
                                         use_bias=False,
                                         kernel_initializer=keras.initializers.he_normal())
        self.bn4 = keras.layers.BatchNormalization(momentum=0.9,
                                                   epsilon=1e-5)
        self.relu4 = keras.layers.ReLU()
        self.maxpool3 = keras.layers.MaxPool2D(pool_size=(2,1), strides=(2,1))
        self.layer3 = self._make_layer(block,
                                       512,
                                       layers[2],
                                       stride=1,
                                       use_gcb=gcb_enabling[2],
                                       gcb_config=gcb_config)


        self.conv5 = keras.layers.Conv2D(512,
                                         kernel_size=3,
                                         padding='same',
                                         use_bias=False,
                                         kernel_initializer=keras.initializers.he_normal())
        self.bn5 = keras.layers.BatchNormalization(momentum=0.9,
                                                   epsilon=1e-5)
        self.relu5 = keras.layers.ReLU()
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=1,
                                       use_gcb=gcb_enabling[3],
                                       gcb_config=gcb_config)


        self.conv6 = keras.layers.Conv2D(512,
                                         kernel_size=3,
                                         padding='same',
                                         use_bias=False,
                                         kernel_initializer=keras.initializers.he_normal())
        self.bn6 = keras.layers.BatchNormalization(momentum=0.9,
                                                   epsilon=1e-5)
        self.relu6 = keras.layers.ReLU()


    def _make_layer(self, block, planes, blocks, stride=1, gcb_config=None, use_gcb=False):
        downsample =None
        if stride!=1 or self.inplanes != planes * block.expansion:

            downsample = keras.Sequential(
                [keras.layers.Conv2D(planes * block.expansion,
                                    kernel_size=(1,1),
                                    strides=stride,
                                    use_bias=False,
                                    kernel_initializer=keras.initializers.he_normal()),
                keras.layers.BatchNormalization(momentum=0.9,
                                                epsilon=1e-5)],
                name='downsample'
            )
        layers = []
        layers.append(block(planes, stride, downsample, gcb_config, use_gcb))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(planes, gcb_config=gcb_config, use_gcb=use_gcb))

        return keras.Sequential(layers, name='make_layer')


    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.layer1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool2(x)
        x = self.layer2(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool3(x)
        x = self.layer3(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.layer4(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        return x