from config import config
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from bert4keras.snippets import DataGenerator, sequence_padding, parallel_apply_generator
from tqdm import tqdm
from models import *

import tensorflow as tf
import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
import glob
from ds_ctcdecoder import UTF8Alphabet
from config import config
from asr_utils.features.specaugment_numpy import Augmentation
augmentation = Augmentation({"time_masking": {"num_masks": 10,
                                                  "mask_factor": 100,
                                                  "p_upperbound": 0.05},
                                 "freq_masking": {"num_masks": 1,
                                                  "mask_factor": 27},
                                 'prob': 0.5})

alphabet = UTF8Alphabet()

# dataset = tf.data.TFRecordDataset(glob.glob('/home/zxa/ps/ASRmodel/data/aidatatang_200zh/tfrecord_audio/test/*'))
# feature_map = {'audio': tf.io.FixedLenFeature((), tf.string),
#                        'label': tf.io.FixedLenFeature((), tf.string),
#                        'filename': tf.io.FixedLenFeature((), tf.string)
#                        }
#
# def audio_gen():
#     for x in tqdm(dataset):
#         parsed_example = tf.io.parse_single_example(x, feature_map)
#         audio = tf.io.decode_raw(parsed_example["audio"], out_type=tf.int16)
#         yield audio.numpy().astype('float32')
#
#
def extract_feature(audio):
    waveform = torch.tensor(audio[None, ...])
    mat = kaldi.fbank(waveform,
                      num_mel_bins=80,
                      frame_length=25,
                      frame_shift=10,
                      dither=0.0,
                      energy_floor=0.0,
                      sample_frequency=16000)
    return mat
#
#
# class data_generator(DataGenerator):
#     """数据生成器
#     """
#     def __init__(self, **kwargs):
#         super(data_generator, self).__init__(**kwargs)
#
#     def __iter__(self, random=False):
#         batch_feature, batch_label, batch_feature_len, batch_label_len = [], [], [], []
#         for i, d in parallel_apply_generator(
#                 func=extract_feature,
#                 iterable=self.sample(random),
#                 workers=1,
#                 max_queue_size=1024
#         ):
#             is_end, feature, target_token_ids = d
#             if config.multi_card:
#                 yield feature, target_token_ids, [np.ceil(np.ceil(feature.shape[0] / 2) / 2)], [len(target_token_ids)]
#             else:
#                 batch_feature.append(feature)
#                 batch_label.append(target_token_ids)
#                 batch_feature_len.append([np.ceil(np.ceil(feature.shape[0] / 2) / 2)])
#                 batch_label_len.append([len(target_token_ids)])
#                 if len(batch_feature) == self.batch_size or is_end:
#                     batch_feature = sequence_padding(batch_feature)
#                     batch_label = sequence_padding(batch_label)
#                     batch_feature_len = sequence_padding(batch_feature_len)
#                     batch_label_len = sequence_padding(batch_label_len)
#                     yield [batch_feature, batch_label, batch_feature_len, batch_label_len], None
#                     batch_feature, batch_label, batch_feature_len, batch_label_len = [], [], [], []
#
#
# dataloader = data_generator(data=audio_gen(), batch_size=64)
# for data in dataloader.__iter__():
#     pass


def tfrecord_dataloader_with_torch(files, train_mode=True, repeat=config.epochs, limit=None, test_mode=False):
    def map_func(example):
        # feature 的属性解析表
        feature_map = {'audio': tf.io.FixedLenFeature((), tf.string),
                               'label': tf.io.FixedLenFeature((), tf.string),
                               'filename': tf.io.FixedLenFeature((), tf.string)
                               }
        parsed_example = tf.io.parse_single_example(example, features=feature_map)

        feature = tf.io.decode_raw(parsed_example["audio"], out_type=tf.int16)
        label = parsed_example["label"]
        return feature, label

    def numpy_process(feature, label):
        label = label.numpy().decode('utf8')
        label = alphabet.Encode(label)
        label = tf.convert_to_tensor(np.array(label))
        label = tf.cast(label, tf.int32)
        label_len = tf.convert_to_tensor([tf.shape(label)[0]])
        feature = tf.convert_to_tensor(extract_feature(feature.numpy().astype('float32')).numpy())
        if train_mode:
            feature = augmentation.augment(np.expand_dims(feature, -1))
            feature = tf.squeeze(feature, -1)
        feature_len = tf.convert_to_tensor([feature.shape[0]])
        return feature, feature_len, label, label_len

    def process_fn(features, feature_len, label, label_len):
        return (features, feature_len, label, label_len), label

    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    if limit:
        dataset = dataset.take(limit)
    # num_parallel_calls会导致数据增强出现随机性，导致每一次结果不一样
    if not test_mode:
        dataset = (dataset.repeat(repeat).map(map_func=map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                   .map(lambda x, y: tf.py_function(numpy_process, inp=[x, y],
                                                         Tout=[tf.float32, tf.int32, tf.int32, tf.int32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                   .padded_batch(config.batch_size, ([None, 80], [1], [None], [1]))
                   .map(process_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                   .prefetch(tf.data.experimental.AUTOTUNE)
                   )
    else:
        dataset = (dataset.map(map_func=map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                   .padded_batch(config.batch_size, ([None, 80], [None], [1], [1]))
                   .map(process_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                   .prefetch(tf.data.experimental.AUTOTUNE)
                   )

    return dataset


if __name__ == '__main__':
    sess = tf.Session()
    files = glob.glob('/home/zxa/ps/ASRmodel/data/aidatatang_200zh/tfrecord_audio/test/*')
    generator = tfrecord_dataloader_with_torch(files, train_mode=False)
    dev_gen_dataset = generator.make_one_shot_iterator()
    iterator = dev_gen_dataset.get_next()
    for i in tqdm(range(5000)):
        dev_data = sess.run(iterator)