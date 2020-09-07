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
# File: model.py
# Author: Owen Lu
# Date: 
# Email: jiangxiluning@gmail.com
# Description:
from typing import *

from easydict import EasyDict
import tensorflow as tf
from tensorflow import keras

from . import backbone
from . import transformer
from . import transformer_tf
from ..dataset import utils

class MasterModel(tf.keras.models.Model):

    def __init__(self, config: EasyDict, vocab_size: int, image_size: tuple):
        super(MasterModel, self).__init__(name='Master')

        self.vocab_size = vocab_size
        self.d_model = config.model_size
        self.headers = config.multiheads
        self.config = config
        self.image_size = image_size

        self.feature_extractor = backbone.Resnet31(backbone.BasicBlock, backbone_config=config.backbone)
        self.seq_embedding = keras.layers.Embedding(vocab_size, self.d_model)
        self.decoder = transformer.Decoder(self.d_model,
                                           self.headers,
                                           position_num=utils.LabelTransformer.max_length,
                                           config=config.decoder)

        # self.decoder = transformer_tf.Decoder(num_layers=config.decoder.stacks,
        #                                       d_model=self.d_model,
        #                                       num_heads=self.headers,
        #                                       dff=config.decoder.feed_forward_size,
        #                                       target_vocab_size=vocab_size,
        #                                       maximum_position_encoding=utils.LabelTransformer.max_length,
        #                                       rate=config.decoder.dropout
        #                                      )
        self.feature_pe = transformer.positional_encoding(image_size[0] * image_size[1], self.d_model)
        self.linear = keras.layers.Dense(vocab_size, kernel_initializer=tf.initializers.he_uniform())

    def get_config(self):
        return { 'config': self.config,
                 'vocab_size': self.vocab_size,
                 'image_size': self.image_size}

    @tf.function
    def make_mask(self, target):
        look_ahead_mask = transformer.create_look_ahead_mask(tf.shape(target)[1])
        target_padding_mask = transformer.create_padding_mask(target, utils.LabelTransformer.dict['<PAD>'])
        combined_mask = tf.maximum(target_padding_mask, look_ahead_mask)
        return None, combined_mask


    def call(self, inputs, **kwargs):
        image: tf.Tensor = inputs[0]
        transcript: tf.Tensor = inputs[1]

        feature = self.feature_extractor(image, **kwargs)

        B = tf.shape(feature)[0]
        H = tf.shape(feature)[1]
        W = tf.shape(feature)[2]
        C = tf.shape(feature)[3]

        feature = tf.reshape(feature, shape=(B, H*W, C))
        memory = feature + self.feature_pe[:, :H*W, :]

        _, tgt_mask = self.make_mask(transcript[:, :-1])

        output = self.decoder(self.seq_embedding(transcript[:, :-1]), memory, None, tgt_mask, **kwargs)
        #output, _ = self.decoder(transcript, memory, training, tgt_mask, None)
        logits = self.linear(output)

        return logits

    # @tf.function
    # def predict(self,
    #             image: tf.Tensor,
    #             training=tf.constant(False)):
    #
    #     feature = self.feature_extractor(image, training=training)
    #     B, H, W, C = feature.shape
    #     feature = tf.reshape(feature, shape=(B, H*W, C))
    #     memory = feature + self.feature_pe[:, :H*W, :]
    #
    #     max_len = tf.constant(utils.LabelTransformer.max_length, dtype=tf.int32)
    #     start_symbol = tf.constant(utils.LabelTransformer.dict['<SOS>'], dtype=tf.int32)
    #
    #     if tf.equal(max_len, tf.constant(-1)):
    #         max_len = tf.constant(100, dtype=tf.int32)
    #     output = tf.fill(dims=(B, 1), value=start_symbol)
    #
    #     final_logits = tf.zeros(shape=(B, max_len - 1, self.vocab_size), dtype=tf.float32)
    #     for i in tf.range(max_len - 1):
    #         tf.autograph.experimental.set_loop_options(shape_invariants=[(output, tf.TensorShape([B, None]))])
    #         _, combined_mask = self.make_mask(output)
    #         logits, _ = self.decoder(output, memory, training, combined_mask, None)
    #         #logits = self.decoder(self.seq_embedding(output), memory, None, combined_mask, training=False)
    #         logits = self.linear(logits)
    #         last_logits = logits[:, -1:, :]
    #
    #         predicted_id = tf.cast(tf.argmax(last_logits, axis=-1), tf.int32)
    #         output = tf.concat([output, predicted_id], axis=-1)
    #
    #         if i == (max_len - 2):
    #             final_logits = logits
    #
    #     return output, final_logits


    @tf.function
    def decode(self,
               image: tf.Tensor,
               ):
        """

        Args:
            image:
            padding:  False is not supported in graph mode

        Returns:

        """

        feature = self.feature_extractor(image, training=False)
        B = tf.shape(feature)[0]
        H = tf.shape(feature)[1]
        W = tf.shape(feature)[2]
        C = tf.shape(feature)[3]
        feature = tf.reshape(feature, shape=(B, H*W, C))
        memory = feature + self.feature_pe[:, :H*W, :]

        max_len = tf.constant(utils.LabelTransformer.max_length, dtype=tf.int32)
        start_symbol = tf.constant(utils.LabelTransformer.dict['<SOS>'], dtype=tf.int32)
        padding_symbol = tf.constant(utils.LabelTransformer.dict['<PAD>'], dtype=tf.int32)

        padding: tf.bool = tf.constant(True)

        if padding:
            ys = tf.fill(dims=(B, max_len - 1), value=padding_symbol)
            start_vector = tf.fill(dims=(B, 1), value=start_symbol)
            ys = tf.concat([start_vector, ys], axis=-1)
        else:
            if tf.equal(max_len, tf.constant(-1)):
                max_len = tf.constant(100, dtype=tf.int32)
            ys = tf.fill(dims=(B, 1), value=start_symbol)

        final_logits = tf.zeros(shape=(B, max_len - 1, self.vocab_size), dtype=tf.float32)
        # max_len = len + 2
        for i in range(max_len - 1):
            tf.autograph.experimental.set_loop_options(shape_invariants=[(final_logits, tf.TensorShape([None, None, self.vocab_size]))])
            _, ys_mask = self.make_mask(ys)
            #output, _ = self.decoder(ys, memory, False, ys_mask, None)
            output = self.decoder(self.seq_embedding(ys), memory, None, ys_mask, training=False)
            logits = self.linear(output)
            prob = tf.nn.softmax(logits, axis=-1)
            next_word = tf.argmax(prob, axis=-1, output_type=ys.dtype)

            if padding:
                # ys.shape = B, T
                i_mesh,j_mesh = tf.meshgrid(tf.range(B), tf.range(max_len), indexing='ij')
                indices = tf.stack([i_mesh[:, i+1], j_mesh[:, i+1]], axis=1)

                ys = tf.tensor_scatter_nd_update(ys, indices, next_word[:, i+1])
            else:
                ys = tf.concat([ys, next_word[:, -1, tf.newaxis]], axis=-1)

            if i == (max_len - 2):
                final_logits = logits

        return ys, final_logits[:, 1:]




