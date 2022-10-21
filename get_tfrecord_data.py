import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import csv
import numpy as np
import wavio
import soundfile as sf
import argparse

from ds_ctcdecoder import UTF8Alphabet
from tqdm import tqdm

alphabet = UTF8Alphabet()

def bytes_feature(value: bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def int64_feature(value: int):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value: float):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def read_wav(dic):
    if '.wav' in dic['wav_filename']:
        read_obj = wavio.read(dic['wav_filename'])
        data = read_obj.data
        rate = read_obj.rate
    elif '.flac' in dic['wav_filename']:
        data, rate = sf.read(dic['wav_filename'], dtype=np.int16)
    assert rate == 16000
    return data, dic


def get_tfrecord_data_from_filterbank(csvpath, savedir):
    from asr_utils.features.speech_featurizers import NumpySpeechFeaturizer
    c = list(csv.DictReader(open(csvpath, 'r', encoding='utf-8')))
    file_count = 1
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    writer = tf.io.TFRecordWriter(savedir + "/%d.tfrecords" % file_count)
    featurizer = NumpySpeechFeaturizer({"preemphasis": 0.97})
    maxlen = 0
    total_file_size = 0
    for idx, line in tqdm(enumerate(c)):
        audio, _ = read_wav(line)
        filename = line['wav_filename']
        feature = featurizer.extract(audio.astype(np.float32))
        feature = np.squeeze(feature)
        label = alphabet.Encode(line['transcript'].replace(' ', ''))
        total_file_size += int(line['wav_filesize']) / 1024 / 1024  # 第一个1024转化为kb，第二个1024转化为m
        if total_file_size >= 200:
            writer.close()
            file_count += 1
            total_file_size = 0
            writer = tf.io.TFRecordWriter(savedir+"/%d.tfrecords" % file_count)
        example = tf.train.Example(features=tf.train.Features(feature={
            'feature': bytes_feature(feature.tobytes()),
            'feature_height': int64_feature(feature.shape[0]),
            'feature_width': int64_feature(feature.shape[1]),
            'label': int64_list_feature(label),
            'label_len': int64_feature(len(label)),
            'filename': bytes_feature(filename.encode('utf-8'))
        }))
        writer.write(example.SerializeToString())
        maxlen = max(feature.shape[0], maxlen)
    writer.close()
    print(f"maxlen {maxlen}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get tfrecord data')
    parser.add_argument('--csv_path',
                        type=str,
                        default=r'D:\open_data\speech\primewords_md_2018_set1\test.csv',
                        help='csv file path')
    parser.add_argument('--save_path',
                        type=str,
                        default=r'D:\open_data\speech\primewords_md_2018_set1\tfrecord_data\test',
                        help='save record file path')
    args = parser.parse_args()
    print(args.is_train)
    get_tfrecord_data_from_filterbank(args.csv_path, args.save_path)