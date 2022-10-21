## GAU transformer
GAU transformer 在语音领域的应用，采用CTC结构+interCTC+stochastic Depth

### 环境
- tensorflow-gpu==1.15.5
- keras==2.3.1
- ds_ctcdecoder==0.9.3
- 其他环境可以在requirement.txt中查找

### 结果
采用primewords_md_2018_set1，wenetspeech—1000h，aidatatang_200zh，magic_data，ST-CMDs，aishell1共同训练了32层GAU模型，共含2000+小时数据，4张2080TI训练12天，共30轮，损失为27.22，对最后6轮的权重进行平均后得到最终模型，在wenet验证集上的结果为11.32。虽然不如wenet的8.80，但是此结构训练所需资源少，且只用了2000+小时数据进行训练

### 模型checkpoint与scorer下载
链接：https://pan.baidu.com/s/1V46Z-n3rW4zvLfNZGOnRBw 
提取码：36r0
### 实时语音识别
`python mic_vad_streaming.py`可以进行实时语音识别，将yamls/evaluate_multitask.yaml中的scorer_path和load_model_path更改为正确的地址

### 文件说明
- train.py: 边生成特征边训练，速度慢，后期没有维护
- train_fast.py: 采用tfrecord格式的数据进行训练，速度快，在服务器上采用多卡训练
- evaluate.py: 用于本地进行预测
- model.py: GAU模型文件
- config.py: 用于加载`yaml/.yaml`文件参数
- optimizer.py: 用于keras的transformer learning rate schedule
- plot_learning_rate.py: 用于绘出transformer learning rate
- asr_utils: 一些用于语音特征提取，语音增强的代码，来自TensorflowASR
- speech_model_interpret.py: 用于实时语音识别
- mic_vad_streaming.py: 用于获取实时语音

### 数据处理
1. 首先是下载数据，用primewords作为例子，primewords可以在[这里](https://www.openslr.org/47/) 下载，其他还可以下载AISHELL1,AIDATATANG,MAGIC_DATA等等，这里以primewords作为教程
2. `python get_csv_data.csv --label_file_path [path1] --audio_dir [path2]`将下载好的primewords的标签文件地址和音频地址作为参数输入后，可以自动获得分割好的训练集，测试集和验证集的csv文件
3. `python get_tfrecord_data.py --csv_path [path1] --save_path [path2]`将处理好的csv文件的地址以及保存tfrecord文件的地址作为参数进行输入，可以自动获得tfrecord数据，需要处理3次，将训练集，测试集，验证集分别运行

### scorer构建[Optional]
预测时可以采用贪婪解码，可以加入语言模型进行beam search的方式进行解码，加入语言模型可以使得预测更加的准确，由于我们采用的是deepspeech的ds_ctcdecoder，因此可以直接使用deepspeech生成的语言模型scorer，生成的scorer的方式如下:
1. 首先下载并安装[kenlm](https://github.com/kpu/kenlm/tree/0c4dd4e8a29a9bcaf22d971a83f4974f1a16d6d9) ，然后在[deepspeech](https://github.com/mozilla/DeepSpeech) 中，将data/lm/generate_lm.py和复制到kenlm目录下，然后下载[这里](https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/native_client.amd64.cpu.linux.tar.xz) ，解压后可以看到`generate_scorer_package`，将该文件也放到kenlm目录下
2. ```
   cd kenlm

   python generate_lm.py --input_txt vocabulary.txt --output_dir . --top_k 500000 --kenlm_bins build/bin/ --arpa_order 5 --max_arpa_memory "85%" --arpa_prune "0|0|1" --binary_a_bits 255 --binary_q_bits 8 --binary_type trie --discount_fallback

   ./generate_scorer_package --alphabet alphabet.txt --lm data/lm.binary --vocab vocab-500000.txt --package kenlm.scorer --default_alpha 1 --default_beta 4 --force_bytes_output_mode True
   ```
需要更改的地方为--input_txt和--alphabet，对应的为文本和最小字符
3. vocabulary.txt的内容如下，以空格分隔每一个字
```
沙 面 宽 阔 平 缓 沙 质 柔 软 细 净 北 端 有 一 巨 石 植 根 沙 间
在 促 成 逃 脱 的 那 种 精 秘 的 微 明 中 常 有 星 光 和 闪 电
项 目 化 管 理 就 是 指 在 一 定 的 条 件 与 资 源 情 况 下
覆 膜 面 料 这 类 面 料 有 点 类 似 于 冲 锋 衣 面 料
经 过 两 年 不 太 愉 快 的 竞 争 在 一 九 零 三 年 两 个 联 盟 达 成 一 个 新 协 议
...
```
alphabet.txt的内容如下
```
沙
面
宽
阔
平
缓
....
```
我在scorers文件夹中放了primewords的scorer，其他数据集的scorer可以通过相同的方式进行生成

### 训练
`python train_fast.py --config_path yamls/primewords_L16_H512_A128.yaml --train`进行单卡训练，设置multi_card=True，gpu_id='0,1,2,3'可以进行任意多卡训练

### 预测
`python evaluate.py --config_path yamls/evaluate_multitask.yaml`，更改yaml中的dev_loaddirs，dev_csvs，load_model_path和scorer_path可以进行预测