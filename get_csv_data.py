import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import re
import glob
import json
import argparse

from pathlib import Path
from sklearn.model_selection import train_test_split
import wave
import collections


def split_and_save2csv(save_dir, res, train_retio=0.9):
    """

    :param save_dir:
    :param res:
    :param train_retio: training set ratio, dev and test has the same ratio
    :return:
    """
    train_res, _ = train_test_split(res, test_size=1-train_retio, random_state=42)
    dev_res, test_res = train_test_split(_, test_size=0.5, random_state=42)
    train_df = pd.DataFrame(
        data=train_res,
        columns=["wav_filename", "wav_filesize", "transcript"],
    )
    dev_df = pd.DataFrame(
        data=dev_res,
        columns=["wav_filename", "wav_filesize", "transcript"],
    )
    test_df = pd.DataFrame(
        data=test_res,
        columns=["wav_filename", "wav_filesize", "transcript"],
    )
    total_df = pd.DataFrame(
        data=res,
        columns=["wav_filename", "wav_filesize", "transcript"],
    )

    train_df.to_csv(os.path.join(save_dir, 'train.csv'), index=False, encoding='utf-8')
    dev_df.to_csv(os.path.join(save_dir, 'dev.csv'), index=False, encoding='utf-8')
    test_df.to_csv(os.path.join(save_dir, 'test.csv'), index=False, encoding='utf-8')
    total_df.to_csv(os.path.join(save_dir, 'total.csv'), index=False, encoding='utf-8')

    print('successfully saved train, dev, test, total csv files to %s\n%s\n%s\n%s\n' % (os.path.join(save_dir, 'train.csv'),
                                                                                        os.path.join(save_dir, 'dev.csv'),
                                                                                        os.path.join(save_dir, 'test.csv'),
                                                                                        os.path.join(save_dir, 'total.csv')))

def process_aishell3():
    def is_cn(word):
        """返回文字是否为中文"""
        cn_pattern = re.compile(r'[\u4e00-\u9fa5]+')
        if cn_pattern.match(word):
            return True
        else:
            return False

    dict = {}
    for i in ['train/content.txt', 'test/content.txt']:
        with open(os.path.join(r'/home/zxa/ps/ASRmodel/data/aishell3', i), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                file_name = line[0]
                word_with_pinyin = line[1].split()
                words = []
                for word in word_with_pinyin:
                    if is_cn(word):
                        words.append(word)
                dict[file_name] = ' '.join(words)

    dir = '/home/zxa/ps/ASRmodel/data/aishell3'
    up_dirs = [os.path.join(dir, 'train/wav'), os.path.join(dir, 'test/wav')]
    res = []
    for up_dir in up_dirs:
        low_dirs = os.listdir(os.path.join(dir, up_dir))
        for low_dir in tqdm(low_dirs):
            low_dir_ = os.path.join(dir, up_dir, low_dir)
            files = os.listdir(low_dir_)
            wav_files = [file for file in files if '.wav' in file]
            wav_files.sort()
            for i in range(len(wav_files)):
                wav_file = wav_files[i]
                filepath = os.path.abspath(os.path.join(low_dir_, wav_file))
                filesize = os.path.getsize(filepath)
                transcript = dict[wav_file]
                res.append((filepath, filesize, transcript))

    total_df = pd.DataFrame(
        data=res,
        columns=["wav_filename", "wav_filesize", "transcript"],
    )
    total_df.to_csv(os.path.join(dir, 'total.csv'), index=False, encoding='utf-8')

    np.random.shuffle(res)
    length = len(res)
    train_res = res[:int(length * 0.9)]
    dev_res = res[int(length * 0.9):int(length * 0.95)]
    test_res = res[int(length * 0.95):int(length * 1)]
    train_df = pd.DataFrame(
        data=train_res,
        columns=["wav_filename", "wav_filesize", "transcript"],
    )
    dev_df = pd.DataFrame(
        data=dev_res,
        columns=["wav_filename", "wav_filesize", "transcript"],
    )
    test_df = pd.DataFrame(
        data=test_res,
        columns=["wav_filename", "wav_filesize", "transcript"],
    )
    train_df.to_csv(os.path.join(dir, 'train.csv'), index=False, encoding='utf-8')
    dev_df.to_csv(os.path.join(dir, 'dev.csv'), index=False, encoding='utf-8')
    test_df.to_csv(os.path.join(dir, 'test.csv'), index=False, encoding='utf-8')


def process_aishell1():
    dict = {}
    with open(r'/home/zxa/ps/ASRmodel/data/data_aishell/transcript/aishell_transcript_v0.8.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            file_name = line[0]
            words = []
            for word in line[1:]:
                for c in word:
                    words.append(c)
            dict[file_name] = ' '.join(words)

    dir = '/home/zxa/ps/ASRmodel/data/data_aishell'
    up_dirs = [os.path.join(dir, 'wav/train'), os.path.join(dir, 'wav/dev'), os.path.join(dir, 'wav/test')]
    total_res = []
    missed = 0
    for up_dir in up_dirs:
        res = []
        files = glob.glob(up_dir+'/*/*.wav')
        for file in tqdm(files):
            filepath = file
            filesize = os.path.getsize(filepath)
            filename = Path(file).name[:-4]
            if filename in dict:
                transcript = dict[filename]
            else:
                missed += 1
                continue
            res.append((filepath, filesize, transcript))
        total_res.extend(res)
        np.random.shuffle(res)
        df = pd.DataFrame(
            data=res,
            columns=["wav_filename", "wav_filesize", "transcript"],
        )
        df.to_csv(os.path.join(dir, Path(up_dir).name+'.csv'), index=False, encoding='utf-8')

    print(f"missed {missed}")
    np.random.shuffle(total_res)
    df = pd.DataFrame(
        data=total_res,
        columns=["wav_filename", "wav_filesize", "transcript"],
    )
    df.to_csv(os.path.join(dir, 'total.csv'), index=False, encoding='utf-8')


def process_librispeech():

    mode = ['train', 'dev-clean', 'dev-other', 'test-clean', 'test-other']
    total_res = []
    for m in mode:
        txt_files = glob.glob(f'/home/zxa/ps/ASRmodel/data/LibriSpeech/{m}*/**/*.txt', recursive=True)
        res = []
        for file in tqdm(txt_files):
            filepath = Path(file)
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.split()
                    filename = line[0] + '.flac'
                    new_path = os.path.join(filepath.parent.as_posix(), filename)
                    filesize = os.path.getsize(new_path)
                    transcript = ' '.join([x.lower() for x in line[1:]])
                    res.append((new_path, filesize, transcript))
        total_res.extend(res)
        np.random.shuffle(res)
        df = pd.DataFrame(
            data=res,
            columns=["wav_filename", "wav_filesize", "transcript"],
        )
        df.to_csv(f'/home/zxa/ps/ASRmodel/data/LibriSpeech/{m}.csv', index=False, encoding='utf-8')

    np.random.shuffle(total_res)
    df = pd.DataFrame(
        data=total_res,
        columns=["wav_filename", "wav_filesize", "transcript"],
    )
    df.to_csv(f'/home/zxa/ps/ASRmodel/data/LibriSpeech/total.csv', index=False, encoding='utf-8')


def process_ljspeech():
    import unicodedata
    import re
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    data_path = '/home/zxa/ps/ASRmodel/data/LJSpeech-1.1/'
    wavs_path = data_path + "/wavs/"
    metadata_path = data_path + "/metadata.csv"

    metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
    metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
    metadata_df = metadata_df[["file_name", "normalized_transcription"]]
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)

    split = int(len(metadata_df) * 0.90)
    df_train = metadata_df[:split]
    df_val = metadata_df[split:]

    print(f"Size of the training set: {len(df_train)}")
    print(f"Size of the training set: {len(df_val)}")

    mode = ['train', 'dev']
    total_res = []
    for m in mode:
        if m == 'train':
            df = df_train
        elif m == 'dev':
            df = df_val
        res = []
        for filename, trans in tqdm(zip(list(df["file_name"]), list(df["normalized_transcription"]))):
            filepath = os.path.join(wavs_path, filename+'.wav')
            filesize = os.path.getsize(filepath)
            trans = trans.lower()
            trans = unicodedata.normalize('NFKD', trans).encode('ascii', 'ignore')
            trans = str(trans, encoding='utf-8')
            trans = re.sub('[^abcdefghijklmnopqrstuvwxyz\'?! ]', '', trans)

            res.append((filepath, filesize, trans))
        total_res.extend(res)
        # np.random.shuffle(res)
        df = pd.DataFrame(
            data=res,
            columns=["wav_filename", "wav_filesize", "transcript"],
        )
        df.to_csv(f'/home/zxa/ps/ASRmodel/data/LJSpeech-1.1/{m}.csv', index=False, encoding='utf-8')

    # np.random.shuffle(total_res)
    df = pd.DataFrame(
        data=total_res,
        columns=["wav_filename", "wav_filesize", "transcript"],
    )
    df.to_csv(f'/home/zxa/ps/ASRmodel/data/LJSpeech-1.1/total.csv', index=False, encoding='utf-8')


def process_primewords(audio_dir, label_file_path):
    dict = {}
    with open(label_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for line in data:
        dict[line['file']] = ' '.join(list(line['text'].replace(' ', '')))
    up_dirs = os.listdir(audio_dir)
    res = []
    for up_dir in up_dirs:
        low_dirs = os.listdir(os.path.join(audio_dir, up_dir))
        for low_dir in tqdm(low_dirs):
            low_dir_ = os.path.join(audio_dir, up_dir, low_dir)
            files = os.listdir(low_dir_)
            wav_files = [file for file in files if '.wav' in file]
            wav_files.sort()
            for i in range(len(wav_files)):
                wav_file = wav_files[i]
                filepath = os.path.abspath(os.path.join(low_dir_, wav_file))
                filesize = os.path.getsize(filepath)
                transcript = dict[wav_file]
                res.append((filepath, filesize, transcript))

    split_and_save2csv(save_dir=Path(audio_dir).parent, res=res, train_retio=0.9)


def process_ST():
    dir = '/home/zxa/ps/ASRmodel/data/ST/ST-CMDS-20170001_1-OS'
    files = os.listdir(dir)
    txt_files = [file for file in files if '.txt' in file]
    wav_files = [file for file in files if '.wav' in file]
    txt_files.sort()
    wav_files.sort()
    res = []
    for i in tqdm(range(len(txt_files))):
        txt_file = txt_files[i]
        wav_file = wav_files[i]
        assert txt_file.split('.')[0] == wav_file.split('.')[0]
        filepath = os.path.abspath(os.path.join(dir, wav_file))
        filesize = os.path.getsize(filepath)
        transcript = ' '.join(list(open(os.path.join(dir, txt_file), 'r', encoding='utf-8').read().strip()))
        res.append((filepath, filesize, transcript))

    total_df = pd.DataFrame(
        data=res,
        columns=["wav_filename", "wav_filesize", "transcript"],
    )
    total_df.to_csv(os.path.join(os.path.dirname(dir), 'total.csv'), index=False, encoding='utf-8')


def process_aidatatang():
    dir = r'/home/zxa/ps/ASRmodel/data/aidatatang_200zh/corpus'
    train_dir = os.path.join(dir, 'train')
    dev_dir = os.path.join(dir, 'dev')
    test_dir = os.path.join(dir, 'test')
    mode = ['train', 'dev', 'test']
    up_dirs = [train_dir, dev_dir, test_dir]
    res = []
    for dir_idx in range(len(up_dirs)):
        file_res = []
        up_dir = up_dirs[dir_idx]
        low_dirs = os.listdir(up_dir)
        for low_dir in tqdm(low_dirs):
            low_dir_ = os.path.join(up_dir, low_dir)
            files = os.listdir(low_dir_)
            txt_files = [file for file in files if '.txt' in file]
            wav_files = [file for file in files if '.wav' in file]
            txt_files.sort()
            wav_files.sort()
            for i in range(len(txt_files)):
                txt_file = txt_files[i]
                wav_file = wav_files[i]
                assert txt_file.split('.')[0] == wav_file.split('.')[0]
                filepath = os.path.abspath(os.path.join(low_dir_, wav_file))
                filesize = os.path.getsize(filepath)
                transcript = ' '.join(list(open(os.path.join(low_dir_, txt_file), 'r', encoding='utf-8').read().strip()))
                res.append((filepath, filesize, transcript))
                file_res.append((filepath, filesize, transcript))
        df = pd.DataFrame(
        data=file_res,
        columns=["wav_filename", "wav_filesize", "transcript"],
    )
        df.to_csv(os.path.join(os.path.dirname(dir), mode[dir_idx]+'.csv'), index=False, encoding='utf-8')

    total_df = pd.DataFrame(
        data=res,
        columns=["wav_filename", "wav_filesize", "transcript"],
    )
    total_df.to_csv(os.path.join(os.path.dirname(dir), 'total.csv'), index=False, encoding='utf-8')


def process_thchs30():
    dir = r'/home/zxa/ps/ASRmodel/data/data_thchs30/'
    mode = ['train', 'dev', 'test']
    txt_dir = os.path.join(dir, 'data')
    total_res = []
    for m in mode:
        res = []
        data_dir = os.path.join(dir, m)
        files = os.listdir(data_dir)
        wav_files = [file for file in files if '.wav' in file and '.trn' not in file and 'scp' not in file]
        for i in tqdm(range(len(wav_files))):
            wav_file = wav_files[i]
            txt_file = wav_file + '.trn'
            filepath = os.path.abspath(os.path.join(data_dir, wav_file))
            filesize = os.path.getsize(filepath)
            transcript = ' '.join(list(open(os.path.join(txt_dir, txt_file), 'r', encoding='utf-8').readline().strip().replace(' ', '')))
            res.append((filepath, filesize, transcript))
        total_res.extend(res)
        df = pd.DataFrame(
            data=res,
            columns=["wav_filename", "wav_filesize", "transcript"],
        )
        df.to_csv(dir + f'/{m}.csv', index=False, encoding='utf-8')
    total_df = pd.DataFrame(
        data=total_res,
        columns=["wav_filename", "wav_filesize", "transcript"],
    )
    total_df.to_csv(dir+'/total.csv', index=False, encoding='utf-8')


def process_magic_data():
    dir = r'/home/zxa/ps/ASRmodel/data/magic_data/'
    train_dir = os.path.join(dir, 'train')
    dev_dir = os.path.join(dir, 'dev')
    test_dir = os.path.join(dir, 'test')
    mode = ['train', 'dev', 'test']
    up_dirs = [train_dir, dev_dir, test_dir]
    res = []
    ignored = 0
    for dir_idx in range(len(up_dirs)):
        file_res = []
        up_dir = up_dirs[dir_idx]
        wav_files = glob.glob(os.path.join(up_dir, '**/*.wav'))
        wav_files = {Path(x).name: x for x in wav_files}
        trans_file = glob.glob(os.path.join(up_dir, '*.txt'))[0]
        trans_df = pd.read_csv(trans_file, header=0, delimiter='\t')
        for idx in tqdm(range(trans_df.shape[0])):
            line = trans_df.iloc[idx]
            if line['UtteranceID'] not in wav_files:
                ignored += 1
                continue
            filepath = wav_files[line['UtteranceID']]
            filesize = os.path.getsize(filepath)
            transcript = ' '.join(line['Transcription'])
            res.append((filepath, filesize, transcript))
            file_res.append((filepath, filesize, transcript))
        df = pd.DataFrame(
            data=file_res,
            columns=["wav_filename", "wav_filesize", "transcript"],
        )
        df.to_csv(os.path.join(os.path.dirname(dir), mode[dir_idx] + '.csv'), index=False, encoding='utf-8')
        print(f"ignored {ignored}")
    total_df = pd.DataFrame(
        data=res,
        columns=["wav_filename", "wav_filesize", "transcript"],
    )
    total_df.to_csv(os.path.join(os.path.dirname(dir), 'total.csv'), index=False, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get tfrecord data')
    parser.add_argument('--label_file_path',
                        type=str,
                        default=r'D:\open_data\speech\primewords_md_2018_set1\set1_transcript.json',
                        help='label file path')
    parser.add_argument('--audio_dir',
                        type=str,
                        default=r'D:\open_data\speech\primewords_md_2018_set1\audio_files',
                        help='audio dir')
    args = parser.parse_args()

    process_primewords(args.audio_dir, args.label_file_path)