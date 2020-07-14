import string
import numpy as np
import cv2
import math
import collections
from PIL import Image
import tensorflow as tf

#keys = '0123456789abcdefghijklmnopqrstuvwxyz `~!@#$%^&*()-_=+{}[]|:;\\/\'",.<>?'
keys = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


class strLabelConverterForTransformer(object):
    """Convert between str and label.
    NOTE:
        Insert `EOS` to the alphabet for attention.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, max_length):
        """

        :param alphabet: keys
        :param max_length:  max_length is mainly for controlling the statistics' stability,
         due to the layer normalisation. and the model can only predict max_length text.
         -1 is for fixing the length, the max_length will be computed dynamically for one batch.
         Empirically, it should be maximum length of text you want to predict.
        """
        self.alphabet = alphabet
        self.dict = {}

        self.dict['<EOS>'] = 1  # start
        self.dict['<SOS>'] = 2
        self.dict['<PAD>'] = 0
        self.dict['<UNK>'] = 3
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i + 4 # 从4开始编码

        self.EOS = self.dict['<EOS>']
        self.SOS = self.dict['<SOS>']
        self.PAD = self.dict['<PAD>']
        self.UNK = self.dict['<UNK>']

        self.nclass = len(self.alphabet) + 4

        assert max_length > 0
        self.max_length = max_length + 2  # len(seq) + <SOS> + <EOS>

        keys_tensor = tf.constant(list(self.dict.values()))
        vals_tensor = tf.constant(list(self.dict.keys()))
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor, key_dtype=tf.int32, value_dtype=tf.string),
            default_value=tf.constant('<UNK>')
        )

    def encode(self, text):
        """对target_label做编码和对齐
        对target txt每个字符串的开始加上SOS，最后加上EOS，并用最长的字符串做对齐
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor targets:max_length × batch_size
        """
        tokens = [self.dict[s] if s in self.alphabet else self.UNK  for s in text]
        target = np.full((self.max_length, ), fill_value=self.PAD)

        target[0] = self.SOS                           # 开始
        target[1:len(tokens) + 1] = tokens
        target[len(tokens) + 1] = self.EOS

        return target

    def decode(self, t):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """

        texts = list(self.dict.keys())[list(self.dict.values()).index(t)]
        return texts

    #@tf.function
    def decode_tensor(self, decoded_tensor: np.ndarray):
        #decoded_tensor = self.table.lookup(decoded_tensor)
        #decoded_tensor = tf.strings.reduce_join(decoded_tensor, axis=-1)

        decoded_transcripts = [''] * decoded_tensor.shape[0]
        for index, char_list in enumerate(decoded_tensor.tolist()):
            transcript = ''
            for c in char_list:
                if c == LabelTransformer.EOS:
                    break
                elif c == LabelTransformer.UNK:
                    continue
                elif c == LabelTransformer.PAD:
                    continue
                elif c == LabelTransformer.SOS:
                    continue
                else:
                    l = LabelTransformer.decode(c)
                    transcript += l
            decoded_transcripts[index] = transcript


        # decoded_tensor = tf.strings.regex_replace(decoded_tensor, '<SOS>', '')
        # decoded_tensor = tf.strings.regex_replace(decoded_tensor, '<PAD>', '')
        # decoded_tensor = tf.strings.regex_replace(decoded_tensor, '<UNK>', '')
        # decoded_tensor = tf.strings.regex_replace(decoded_tensor, '<EOS>', '')
        return decoded_transcripts




LabelTransformer =  strLabelConverterForTransformer(keys, max_length=50)

def get_vocabulary(voc_type, SOS='SOS', EOS='EOS', PADDING='PAD', UNKNOWN='UNK'):

    voc = None
    types = ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    if voc_type == 'LOWERCASE':
        voc = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'ALLCASES':
        voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
        voc = list(string.printable[:-6])
    else:
        raise KeyError('voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')

    # update the voc with specifical chars
    voc.append(SOS)
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))

    return voc, char2id, id2char

def rotate_img(img, angle, scale=1):
    H, W, _ = img.shape
    rangle = np.deg2rad(angle)  # angle in radians
    new_width = (abs(np.sin(rangle) * H) + abs(np.cos(rangle) * W)) * scale
    new_height = (abs(np.cos(rangle) * H) + abs(np.sin(rangle) * W)) * scale

    rot_mat = cv2.getRotationMatrix2D((new_width * 0.5, new_height * 0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(new_width - W) * 0.5, (new_height - H) * 0.5, 0]))
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]

    rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(new_width)), int(math.ceil(new_height))),
                             flags=cv2.INTER_LANCZOS4)

    return rot_img

def resize_width(img:tf.Tensor, label:tf.Tensor, width, height, interpolation=tf.image.ResizeMethod.BILINEAR):
    img = tf.image.resize_with_pad(img,
                                   target_height=height,
                                   target_width=width,
                                   method=interpolation)
    img = img/255
    img = (img - 0.5) / 0.5
    return img, label