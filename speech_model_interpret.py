import os
import numpy as np
from ds_ctcdecoder import Scorer, UTF8Alphabet, ctc_beam_search_decoder_batch, ctc_beam_search_decoder
import time
from asr_utils.features.speech_featurizers import NumpySpeechFeaturizer
from train_fast import model
from config import config


model.load_weights(config.load_model_path)
alphabet = UTF8Alphabet()
scorer = Scorer(1, 4, scorer_path=config.scorer_path, alphabet=UTF8Alphabet())
featurizer = NumpySpeechFeaturizer({"preemphasis": 0.97})
def convert_to_model_input(bytes):
    audio = np.frombuffer(buffer=bytes, dtype=np.int16)
    feature = featurizer.extract(audio)
    feature = feature.astype(np.float32)
    feature = np.squeeze(feature)
    feature = np.expand_dims(feature, 0)
    return feature


def pad(inputs):
    append = np.zeros((1, config.feature_len - inputs.shape[1], config.feature_dim))
    feature = np.concatenate((inputs, append), axis=1)
    return feature

def predict_speech(inputs):
    start = time.time()
    input_data = convert_to_model_input(inputs)

    length = input_data.shape[1]
    length = [np.ceil(np.ceil(length / 2) / 2)]
    input_data = pad(input_data)
    logits = model.predict((input_data, np.ones(1), np.ones((1,1)), np.ones(1)))[0]
    decoded1 = ctc_beam_search_decoder(logits.tolist(),
                                             alphabet, 100, scorer=scorer)
    print(decoded1[0][1], time.time()-start)


if __name__ == '__main__':
    import wavio
    import glob
    files = glob.glob(r'D:\open_data\speech\data_virtual\wav\2021_11_30\*.wav')
    for file in files:
        data = wavio.read(file).data.tobytes()
        predict_speech(data)


