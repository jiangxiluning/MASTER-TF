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
# File: eval_iiit5k.py
# Author: Owen Lu
# Date: 
# Email: jiangxiluning@gmail.com
# Description:
from typing import *
import time

import click
import tensorflow as tf
import tqdm
import lmdb
import numpy as np
import cv2
import anyconfig
from easydict import EasyDict

from src.tools.eval_net import Predictor
from src.dataset.utils import LabelTransformer

@click.command()
@click.option('--ckpt', help='checkpoint', type=click.Path(), required=True)
@click.option('--cfg', help='model config', type=click.Path(exists=True), required=True)
@click.option('--output', '-o', help='results output dir', required=True)
@click.option('--iiit5k', '-i', help='iiit5k test lmdb dataset', type=click.Path(exists=True), required=True)
def main(ckpt:str, cfg:str, output:str, iiit5k:str):
    env = lmdb.open(iiit5k, readonly=True, max_readers=32)

    config = anyconfig.load(cfg, ac_parser='yaml')
    config = EasyDict(config)
    predictor = Predictor(ckpt=ckpt, config=config, img_width=160, img_height=48)

    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))

        times = []
        images = []
        predicted_transcripts = []
        transcripts = []
        corrected = 0

        for index in tqdm.trange(nSamples):
            index += 1
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')

            if len(label) > ( LabelTransformer.max_length - 2):
                print('{} is longer than {}, ignored!'.format(label, LabelTransformer.max_length))
                continue

            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            buf = np.frombuffer(imgbuf, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            img = img[np.newaxis, :,:, np.newaxis]

            images.append(img[0])
            transcripts.append(label)
            img = tf.constant(img)

            start = time.time()
            pred = predictor.predict(img)
            pred = pred[0]
            times.append(time.time() - start)
            predicted_transcripts.append(pred)

            if pred.lower() == label.lower(): # case insensitive
                corrected += 1


        print('Word Accuracy: {} '.format(corrected/len(transcripts)))
        print('{} samples done with {} FPS.'.format(nSamples, nSamples/(sum(times))))

if __name__ == '__main__':
    main()