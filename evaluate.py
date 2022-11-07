from train_fast import model, tf, config, tqdm, keras, alphabet, build_model, average_checkpoints, Scorer
from train_fast import ctc_beam_search_decoder_batch, np, check_files, tfrecord_dataloader, cal_steps
from jiwer import wer

# 加载训练用的tfrecord数据文件
dev_files = check_files(config.dev_loaddirs)

# 数据生成器
dev_gen_dataset = tfrecord_dataloader(dev_files, train_mode=False)
dev_steps = cal_steps(config.dev_csvs, config.batch_size)

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
    if config.num_checkpoints == 1:
        model.load_weights(config.load_model_path)
    else:
        model = average_checkpoints()
        model.save_weights('best_model.h5')

    scorer = Scorer(1, 4, scorer_path=config.scorer_path, alphabet=alphabet)
    pbar = tqdm(range(dev_steps))
    cers = []
    try:
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
    except tf.errors.OutOfRangeError:
        print('end! some value is filtered in dataloader')
    print(np.mean(cers), true_labels[0], pred_labels[0])