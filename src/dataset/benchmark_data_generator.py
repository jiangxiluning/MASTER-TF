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
# File: benchmark_data_generator.py
# Author: Owen Lu
# Date: 
# Email: jiangxiluning@gmail.com
# Description:
from typing import *
import csv
from pathlib import Path
import re

import cv2
from loguru import logger
import lmdb
import numpy as np

from .utils import LabelTransformer
from .utils import keys

def generator_folder(folder: str):
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError('{} is not exists.'.format(folder_path.as_posix()))

    image_folder = folder_path.joinpath('images')
    gt_file = folder_path.joinpath('gt.tsv')

    if not image_folder.exists():
        raise FileNotFoundError('{} is not exists.'.format(image_folder.as_posix()))

    if not gt_file.exists():
        raise FileNotFoundError('{} is not exists.'.format(gt_file.as_posix()))

    with open(gt_file, encoding='utf8') as f:
        try:
            csv_file = csv.reader(f, delimiter='\t')
            for index, row in enumerate(csv_file):
                image_file = row[0]
                transcript = row[1]

                img = cv2.imread(image_folder.joinpath(image_file).as_posix(), cv2.IMREAD_GRAYSCALE)
                img = img.unsqueeze(-1)

                if len(transcript) > (LabelTransformer.max_length - 2):
                    logger.warning('{} is longer than {}, ignored!'.format(word, LabelTransformer.max_length))
                    continue
                word = LabelTransformer.encode(transcript)

                yield img, word
        except Exception as e:
            logger.exception("Error in {}".format(image_folder.joinpath(image_file).as_posix()))
            raise e


def generator_lmdb(lmdb_dir:str, rgb=False):

    env = lmdb.open(lmdb_dir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    if not env:
        logger.exception('cannot create lmdb from %s' % (lmdb_dir))
        raise RuntimeError('cannot create lmdb from %s' % (lmdb_dir))

    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))
        for index in range(nSamples):
            index += 1  # lmdb starts with 1
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')

            if len(label) > (LabelTransformer.max_length - 2):
                # print(f'The length of the label is longer than max_length: length
                # {len(label)}, {label} in dataset {self.root}')
                logger.warning('{} is longer than {}, ignored!'.format(label, LabelTransformer.max_length))
                continue

            # By default, images containing characters which are not in opt.character are filtered.
            # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
            out_of_char = f'[^{keys}]'
            if re.search(out_of_char, label.lower()):
                continue

            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            try:
                if rgb:
                    buf = np.frombuffer(imgbuf, dtype=np.uint8)
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                else:
                    buf = np.frombuffer(imgbuf, dtype=np.uint8)
                    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
                img = img[:,:, np.newaxis]

            except Exception as e:
                logger.exception(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                raise e

            word = LabelTransformer.encode(label)
            yield img, word













