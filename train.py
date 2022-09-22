from config import config
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from bert4keras.snippets import DataGenerator, sequence_padding, parallel_apply_generator
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from tqdm import tqdm
from models import *
from pathlib import Path
from datetime import datetime
from asr_utils.features.speech_featurizers import NumpySpeechFeaturizer
from asr_utils.features.specaugment_numpy import Augmentation
from ds_ctcdecoder import UTF8Alphabet
from jiwer import wer

import tensorflow as tf
import numpy as np
import random
import pandas as pd
import soundfile as sf
import wavio
np.random.seed(config.seed)
tf.set_random_seed(config.seed)
random.seed(config.seed)

if config.multi_card:
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

alphabet = UTF8Alphabet()
featurizer = NumpySpeechFeaturizer({"preemphasis": 0.97})
augmentation = Augmentation({"time_masking": {"num_masks": 10,
                                                  "mask_factor": 100,
                                                  "p_upperbound": 0.05},
                                 "freq_masking": {"num_masks": 1,
                                                  "mask_factor": 27},
                                 'prob': 0.5})


def read_wav(dataset):
    while True:
        for d in dataset:
            filename, target = d
            if '.wav' in filename:
                read_obj = wavio.read(filename)
                data = read_obj.data
                rate = read_obj.rate
            elif '.flac' in filename:
                data, rate = sf.read(filename, dtype=np.int16)
            target = alphabet.Encode(target.replace(' ', ''))
            assert rate == 16000
            yield data, target


def train_encode(is_end_audio):
    is_end, (audio, target) = is_end_audio
    feature = featurizer.extract(audio.astype(np.float32))
    feature = augmentation.augment(feature)
    feature = np.squeeze(feature, axis=-1)
    return is_end, feature, target


def dev_encode(is_end_audio):
    is_end, (audio, target) = is_end_audio
    feature = featurizer.extract(audio.astype(np.float32))
    return is_end, feature, target


class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self, run_mode=None, **kwargs):
        super(data_generator, self).__init__(**kwargs)
        if run_mode == 'train':
            self.encode = train_encode
        else:
            self.encode = dev_encode

    def __iter__(self, random=False):
        batch_feature, batch_label, batch_feature_len, batch_label_len = [], [], [], []
        for i, d in parallel_apply_generator(
                func=self.encode,
                iterable=self.sample(random),
                workers=4,
                max_queue_size=1024
        ):
            is_end, feature, target_token_ids = d
            if config.multi_card:
                yield feature, target_token_ids, [np.ceil(np.ceil(feature.shape[0] / 2) / 2)], [len(target_token_ids)]
            else:
                batch_feature.append(feature)
                batch_label.append(target_token_ids)
                batch_feature_len.append([np.ceil(np.ceil(feature.shape[0] / 2) / 2)])
                batch_label_len.append([len(target_token_ids)])
                if len(batch_feature) == self.batch_size or is_end:
                    batch_feature = sequence_padding(batch_feature)
                    batch_label = sequence_padding(batch_label)
                    batch_feature_len = sequence_padding(batch_feature_len)
                    batch_label_len = sequence_padding(batch_label_len)
                    yield [batch_feature, batch_label, batch_feature_len, batch_label_len], None
                    batch_feature, batch_label, batch_feature_len, batch_label_len = [], [], [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred, y_len, label_len = inputs
        loss = K.ctc_batch_cost(y_true, y_pred, y_len, label_len)

        return loss


def build_model():
    # 词表不同，将Encoder和Decoder分开创建
    MODEL = GAU_alpha(
        vocab_size=None,
        hidden_size=config.hidden_size, # 512
        num_hidden_layers=config.n_layer, # 16
        num_attention_heads=config.n_head, # 1
        attention_key_size=config.attention_key_size,
        intermediate_size=config.inter_hidden_size, # 1024
        hidden_act=config.hidden_act, # 'swish'
        dropout_rate=config.dropout_rate, # 0.1
        attention_dropout_rate=config.attention_dropout_rate, # 0.1
        max_position=512
    )
    MODEL.build()

    output = MODEL.call(MODEL.inputs)
    y_true = Input(shape=(None, ), name='y_true')
    y_len = Input(shape=(1, ), name='y_len')
    label_len = Input(shape=(1, ), name='label_len')
    outputs = CrossEntropy([0])([y_true, output, y_len, label_len])
    model = keras.models.Model(MODEL.inputs + [y_true, y_len, label_len], outputs)

    # 线性递增递减学习率
    Adamp = extend_with_piecewise_linear_lr(Adam)
    model.compile(optimizer=Adamp(learning_rate=config.learning_rate,
                                  lr_schedule={0: 0.0, config.warmup:1., 400000: 0.1},
                                  ))
    model.summary()
    return model


strategy = tf.distribute.MirroredStrategy()
if config.multi_card:
    with strategy.scope():
        model = build_model()
else:
    model = build_model()

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    from https://github.com/ZhuiyiTechnology/t5-pegasus/blob/main/finetune.py
    """
    def on_epoch_end(self, epoch, logs=None):
        model.save_weights(config.save_model_path)  # 保存模型


def load_data(filename):
    """加载数据，一行一条文本"""
    maxlen = 0
    max_filesize = 0
    out = []
    df = pd.read_csv(filename, header=0)
    for i in range(df.shape[0]):
        wav_filename = df.loc[i, 'wav_filename']
        wav_filesize = df.loc[i, 'wav_filesize']
        transcript = df.loc[i, 'transcript']
        maxlen = max(maxlen, len(transcript.split()))
        max_filesize = max(max_filesize, wav_filesize)
        if wav_filesize > 640044:  # 去掉大于20s的
            continue
        out.append((wav_filename, transcript))
    print(f"maxlen of {filename} is {maxlen}, max_filesize of {filename} is {max_filesize}, number of samples is {len(out)}")
    return out

if __name__ == '__main__':
    print(config.__dict__)
    if not os.path.exists(Path(config.save_model_path).parent):
        os.mkdir(Path(config.save_model_path).parent)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    data = load_data(config.train_csvs[0])
    valid_data = load_data(config.dev_csvs[0])

    train_loader = data_generator(data=read_wav(data), batch_size=config.batch_size, run_mode='train')  # 只有训练集需要加mode=train，加入数据增强
    for data in tqdm(train_loader.__iter__()):
        pass
    valid_loader = data_generator(data=read_wav(valid_data), batch_size=config.batch_size)
    evaluator = Evaluator()
    if config.continue_train:
        model.load_weights(config.save_model_path)
    csv_logger = keras.callbacks.CSVLogger(f"{config.log_dir}train_{config.task_name}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(config.save_model_path,
                                                                save_weights_only=True,
                                                                monitor='loss', save_best_only=True)

    steps = len(data) // config.batch_size
    if len(data) % config.batch_size != 0:
        steps += 1
    if config.train_mode:
        if not config.multi_card:
            if config.belu_callback:
                model.fit(train_loader.forfit(), epochs=config.epochs, steps_per_epoch=steps,
                                  callbacks=[evaluator, csv_logger], initial_epoch=config.initial_epoch,
                          use_multiprocessing=False, workers=0)
            else:
                model.fit(train_loader.forfit(), epochs=config.epochs, steps_per_epoch=len(train_loader),
                                  validation_data=valid_loader.forfit(), validation_steps=len(valid_loader),
                                  callbacks=[model_checkpoint_callback, csv_logger], initial_epoch=config.initial_epoch)
        else:
            train_dataset = train_loader.to_dataset(
                types=('float32', 'int32', 'int32', 'int32'),
                shapes=([None, 80], [None], [1], [1]),
                names=('Encoder-Input-Feature', 'y_true', 'y_len', 'label_len'),
                padded_batch=True
            )
            valid_dataset = valid_loader.to_dataset(
                types=('float32', 'int32', 'int32', 'int32'),
                shapes=([None, 80], [None], [1], [1]),
                names=('Encoder-Input-Feature', 'y_true', 'y_len', 'label_len'),
                padded_batch=True
            )
            model.fit(train_dataset, epochs=config.epochs, steps_per_epoch=steps,
                              callbacks=[evaluator, csv_logger])
    else:
        test_data = load_data(config.test_dir)
        model.load_weights(config.save_model_path)
        print(evaluator.evaluate(test_data))
else:
    model.load_weights(config.save_model_path)
