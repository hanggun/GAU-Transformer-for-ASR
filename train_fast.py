from config import config
import os
os.environ['TF_KERAS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

from bert4keras.snippets import DataGenerator, sequence_padding, parallel_apply_generator
from bert4keras.optimizers import Adam
from optimizer import extend_with_transformer_schedule
from tqdm import tqdm
from models import *
from pathlib import Path
from datetime import datetime
from asr_utils.features.augmentation import Augmentation
from ds_ctcdecoder import UTF8Alphabet, Scorer, ctc_beam_search_decoder_batch, Alphabet
import csv

import tensorflow as tf
import numpy as np
import random
import pandas as pd
import glob

np.random.seed(config.seed)
tf.set_random_seed(config.seed)
# tf.random.set_seed(config.seed)
random.seed(config.seed)

if config.multi_card:
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

alphabet = UTF8Alphabet()
# alphabet = Alphabet('/home/zxa/ps/ASRmodel/data/primewords_md_2018_set1/alphabet.txt')
augmentation = Augmentation({"feature_augment": {"time_masking": {"num_masks": 10,
                                                                  "mask_factor": 100,
                                                                  "p_upperbound": 0.05},
                                                 "freq_masking": {"num_masks": 1,
                                                                  "mask_factor": 27}},
                                                 "prob": 0.5})


def tfrecord_dataloader(files, train_mode=True, repeat=config.epochs, limit=None):
    def map_func(example):
        # feature 的属性解析表
        feature_map = {'feature': tf.io.FixedLenFeature((), tf.string),
                       'feature_height': tf.io.FixedLenFeature((), tf.int64),
                       'feature_width': tf.io.FixedLenFeature((), tf.int64),
                       'label': tf.io.VarLenFeature(tf.int64),
                       }
        parsed_example = tf.io.parse_single_example(example, features=feature_map)

        feature = tf.io.decode_raw(parsed_example["feature"], out_type=tf.float32)
        height = parsed_example["feature_height"]
        width = parsed_example["feature_width"]
        label = tf.sparse.to_dense(parsed_example["label"])
        label_len = tf.shape(label)[0]

        feature = tf.reshape(feature, [height, width])
        return feature, label, tf.math.ceil(tf.math.ceil(height[..., None]/2)/2), label_len[..., None]

    def augment(features, transcripts, seq_lens, label_lens):
        features = augmentation.feature_augment(tf.expand_dims(features, -1))
        features = tf.squeeze(features, -1)
        return features, transcripts, seq_lens, label_lens

    def process_fn(features, transcripts, seq_lens, label_lens):
        features = tf.cast(features, tf.float32)
        seq_lens = tf.cast(seq_lens, tf.int32)
        transcripts = tf.cast(transcripts, tf.int32)
        label_lens = tf.cast(label_lens, tf.int32)
        return (features, transcripts, seq_lens, label_lens), transcripts

    def filter_fn(features, transcripts, seq_lens, label_lens):
        return tf.logical_and(
            tf.less_equal(seq_lens, tf.constant([config.feature_len//4], dtype=tf.float64)),
            tf.less(tf.cast(label_lens+5, dtype=tf.float64), seq_lens)
        )[0]

    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    if limit:
        dataset = dataset.take(limit)
    # num_parallel_calls会导致数据增强出现随机性，导致每一次结果不一样
    if train_mode:
        dataset = (dataset.repeat(repeat).map(map_func=map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                   .filter(filter_fn)
                   .map(map_func=augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                   .padded_batch(config.batch_size, ([config.feature_len, 80], [None], [1], [1]))
                   .map(process_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                   .prefetch(tf.data.experimental.AUTOTUNE)
                   )
    else:
        dataset = (dataset.map(map_func=map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                   .padded_batch(config.batch_size, ([config.feature_len, 80], [None], [1], [1]))
                   .map(process_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                   .prefetch(tf.data.experimental.AUTOTUNE)
                   )

    return dataset


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        if not config.inter_CTC:
            y_true, y_pred, y_len, label_len = inputs
            loss = K.ctc_batch_cost(y_true, y_pred, y_len, label_len)
        else:
            y_true, y_pred, y_len, label_len, y_inter_pred = inputs
            loss = K.ctc_batch_cost(y_true, y_pred, y_len, label_len)
            inter_loss = K.ctc_batch_cost(y_true, y_inter_pred, y_len, label_len)
            loss = 0.7 * loss + 0.3 * inter_loss

        return loss


def build_model():
    # 词表不同，将Encoder和Decoder分开创建
    MODEL = GAU_alphaV2(
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
    gau_model = MODEL.model
    y_true = Input(shape=(None, ), name='y_true')
    y_len = Input(shape=(1, ), name='y_len')
    label_len = Input(shape=(1, ), name='label_len')

    final_output = Dense(units=alphabet.GetSize()+1,
                        use_bias=True,
                        kernel_initializer=MODEL.initializer,
                        activation='softmax',
                        name='CTC-Layer')(gau_model.output)

    if config.inter_CTC:
        inter_output = gau_model.get_layer('Transformer-%d-GatedAttentionUnit-Norm' % (config.n_layer//2)).output
        inter_output = Dense(units=2048,
                            use_bias=True,
                            kernel_initializer=MODEL.initializer,
                            activation='relu',
                            name='Inter-Projection')(inter_output)
        inter_output = Dense(units=alphabet.GetSize()+1,
                            use_bias=True,
                            kernel_initializer=MODEL.initializer,
                            activation='softmax',
                            name='Inter-CTC-Layer')(inter_output)

    if not config.inter_CTC:
        outputs = CrossEntropy([1])([y_true, final_output, y_len, label_len])
    else:
        outputs = CrossEntropy([1])([y_true, final_output, y_len, label_len, inter_output])
    model = keras.models.Model(gau_model.inputs + [y_true, y_len, label_len], outputs)
    # 线性递增递减学习率
    Adamt = extend_with_transformer_schedule(Adam)
    model.compile(optimizer=Adamt(learning_rate=config.learning_rate,
                                  start_step=config.start_step,
                                  warmup_steps=config.warmup,
                                  d_model=config.hidden_size,
                                  beta_1=0.9,
                                  beta_2=0.98
                                  ))
    return model


strategy = tf.distribute.MirroredStrategy()
if config.multi_card:
    with strategy.scope():
        model = build_model()
else:
    model = build_model()

model.summary()


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    from https://github.com/ZhuiyiTechnology/t5-pegasus/blob/main/finetune.py
    """
    def __init__(self):
        super().__init__()
        self.count = 0
    def on_epoch_end(self, epoch, logs=None):
        self.count %= 10
        model.save_weights(config.save_model_path + str(self.count)+'.h5')
        print(f'count {self.count}')
        self.count += 1


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


def check_files(loaddirs):
    files = []
    for loaddir in loaddirs:
        tmp = glob.glob(loaddir + '/*.tfrecords')
        if tmp:
            files.extend(tmp)
        else:
            raise FileExistsError(f"no file exists in loaddir")
    return files


def cal_steps(csv_files, batch_size):
    total = 0
    for file in csv_files:
        data = list(csv.DictReader(open(file, 'r', encoding='utf-8')))
        if config.max_file_size:
            data = [x for x in data if int(x['wav_filesize']) <= config.max_file_size]
        total += len(data)
    steps = total // batch_size if total % batch_size == 0 else total // batch_size + 1
    print('number of samples is %d' % total)
    return steps


if __name__ == '__main__':
    print(config.__dict__)
    if not os.path.exists(Path(config.save_model_path).parent):
        os.mkdir(Path(config.save_model_path).parent)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)

    train_files = check_files(config.train_loaddirs)
    dev_files = check_files(config.dev_loaddirs)

    gen_dataset = tfrecord_dataloader(train_files, train_mode=True)
    dev_gen_dataset = tfrecord_dataloader(dev_files, train_mode=False)

    train_steps = cal_steps(config.train_csvs, config.batch_size)
    dev_steps = cal_steps(config.dev_csvs, config.batch_size)

    evaluator = Evaluator()
    if config.continue_train:
        model.load_weights(config.load_model_path)
    if config.multi_card:
        csv_logger = keras.callbacks.CSVLogger(f"{config.log_dir}train_{config.task_name}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")
    else:
        csv_logger = tf.keras.callbacks.CSVLogger(f"{config.log_dir}train_{config.task_name}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(config.save_model_path,
                                                                save_weights_only=True,
                                                                monitor='loss', save_best_only=True)

    if config.train_mode:
        if config.belu_callback:
            model.fit(gen_dataset, epochs=config.epochs, steps_per_epoch=train_steps,
                              callbacks=[evaluator, csv_logger])
    else:
        from jiwer import wer
        dev_gen_dataset = dev_gen_dataset.make_one_shot_iterator()
        iterator = dev_gen_dataset.get_next()
        sess = tf.Session()
        true_labels = []
        pred_labels = []
        if config.greedy:
            model.load_weights(config.load_model_path)
            for _ in tqdm(range(dev_steps)):
                dev_data = sess.run(iterator)
                pred = model.predict(dev_data[0])
                results = sess.run(keras.backend.ctc_decode(pred, input_length=tf.squeeze(dev_data[0][2], -1), greedy=True)[0][0])
                for _true, _pred in zip(dev_data[0][1].tolist(), results.tolist()):
                    _true = [x for x in _true if x != 0]
                    _pred = [x for x in _pred if x != -1]
                    true_labels.append(' '.join(alphabet.Decode(_true)))
                    pred_labels.append(' '.join(alphabet.Decode(_pred)))
            cer = wer(true_labels, pred_labels)
            print(cer)
        else:
            model = build_model()
            weights = []
            for i in tqdm(range(config.num_checkpoints), desc='load model'):
                if not weights:
                    model.load_weights('%s%s.h5' % (config.save_model_path,str(i)))
                    weights = model.get_weights()
                else:
                    model = build_model()
                    model.load_weights('%s%s.h5' % (config.save_model_path,str(i)))
                    tmp_weights = model.get_weights()
                    for i, w in enumerate(tmp_weights):
                        weights[i] += w
            for i in range(len(weights)):
                weights[i] = weights[i] / config.num_checkpoints
            model = build_model()
            model.set_weights(weights)

            scorer = Scorer(1, 4, scorer_path=config.scorer_path, alphabet=alphabet)
            pbar = tqdm(range(dev_steps))
            cers = []
            for _ in pbar:
                dev_data = sess.run(iterator)
                pred = model.predict(dev_data[0])
                decoded = ctc_beam_search_decoder_batch(pred.tolist(), np.squeeze(dev_data[0][2]).tolist(),
                                                             alphabet, 100,
                                                             num_processes=12, cutoff_top_n=40, scorer=scorer)
                for _true, _pred in zip(dev_data[0][1].tolist(), decoded):
                    _true = [x for x in _true if x != 0]
                    true_labels.append(' '.join(alphabet.Decode(_true)))
                    pred_labels.append(' '.join(_pred[0][1]))
                cers.append(wer(true_labels, pred_labels))
                pbar.set_description('cer: %.4f' % np.mean(cers))
            print(np.mean(cers), true_labels[0], pred_labels[0])
else:
    model.load_weights(config.save_model_path)
