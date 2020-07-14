import time

import lmdb
import numpy as np
import cv2
from loguru import logger

from .generator_enqueuer import GeneratorEnqueuer
from .utils import get_vocabulary, rotate_img, LabelTransformer


def generator(lmdb_dir, rgb=False):

    env = lmdb.open(lmdb_dir, max_readers=32, readonly=True)
    txn = env.begin()

    num_samples = int(txn.get(b"nSamples").decode())
    index = np.arange(0, num_samples) # TODO check index is reliable

    for i in index:
        i += 1
        try:
            image_key = 'image-{}'.format(i).encode()
            label_key = 'transcript-{}'.format(i).encode()

            imgbuf = txn.get(image_key)
            imgbuf = np.frombuffer(imgbuf, dtype=np.uint8)

            try:
                if rgb:
                    img = cv2.imdecode(imgbuf, cv2.IMREAD_COLOR)
                else:
                    img = cv2.imdecode(imgbuf, cv2.IMREAD_GRAYSCALE)
            except Exception as e:
                logger.exception(f'Corrupted image for {i}')
                raise e

            img = img[:,:, np.newaxis]
            word = txn.get(label_key).decode()
            if len(word) > (LabelTransformer.max_length - 2):
                logger.warning('{} is longer than {}, ignored!'.format(word, LabelTransformer.max_length))
                continue
            label = LabelTransformer.encode(word)
            yield img, label

        except Exception as e:
            logger.exception("Error in %d" % i)
            continue
    env.close()