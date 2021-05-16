#! -*- coding: utf-8 -*-

import json
import numpy as np
import json
from tqdm import tqdm
import os, sys
sys.path.append(os.getcwd())
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_weight_decay
from bert4keras.optimizers import extend_with_gradient_accumulation
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
import test4nlp.chatbot_seq2seq.config.chatbot_seq2seq_config as config

def corpus(data_path):
    """循环读取语料
    """
    while True:
        ls = json.load(open(config.data_path, encoding='utf-8'))
        print(len(ls))
        for l in ls:
            yield l


# 加载并精简词表
token_dict, keep_tokens = load_vocab(
    dict_path=config.dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)

# 补充词表
compound_tokens = []
for l in open(config.use_tokens_path, encoding='utf-8'):
    token, count = l.strip().split('\t')
    if int(count) >= 10 and token not in token_dict:
        token_dict[token] = len(token_dict)
        compound_tokens.append([0])

# 建立分词器
tokenizer = Tokenizer(token_dict, do_lower_case=True)
print(tokenizer._token_start_id)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, texts in self.sample(random):
            token_ids, segment_ids = [tokenizer._token_start_id], [0]
            for i, text in enumerate(texts):
                ids = tokenizer.encode(text)[0][1:]
                if len(token_ids) + len(ids) <= config.maxlen:
                    token_ids.extend(ids)
                    segment_ids.extend([i % 2] * len(ids))
                else:
                    break
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉padding部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(mask[1], K.floatx())[:, 1:]
        y_true = y_true[:, 1:]  # 目标token_ids
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config.config_path,
    config.checkpoint_path,
    model='nezha',
    application='lm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    compound_tokens=compound_tokens,  # 要扩充的词表
)

output = CrossEntropy(1)([model.inputs[0], model.outputs[0]])

model = Model(model.inputs, output)

AdamW = extend_with_weight_decay(Adam, 'AdamW')
AdamWG = extend_with_gradient_accumulation(AdamW, 'AdamWG')
optimizer = AdamWG(
    learning_rate=2e-5,
    weight_decay_rate=0.01,
    exclude_from_weight_decay=['Norm', 'bias'],
    grad_accum_steps=16
)
model.compile(optimizer=optimizer)


class ChatBot(AutoRegressiveDecoder):
    """基于随机采样对话机器人
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        curr_segment_ids = np.ones_like(output_ids) - segment_ids[0, -1]
        segment_ids = np.concatenate([segment_ids, curr_segment_ids], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def response(self, texts, topk=5):
        token_ids, segment_ids = [tokenizer._token_start_id], [0]
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)[0][1:]
            token_ids.extend(ids)
            segment_ids.extend([i % 2] * len(ids))
        results = self.random_sample([token_ids, segment_ids], 1, topk)
        return tokenizer.decode(results[0])


chatbot = ChatBot(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)


class Evaluator(keras.callbacks.Callback):
    """保存模型权重
    """
    def __init__(self):
        self.lower = 1e10

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] <= self.lower:
            self.lower = logs['loss']
            while True:
                try:
                    model.save_weights(config.save_path)
                    break
                except:
                    print('保存失败，正在重试。。。')
        # while True:
        #     try:
        #         model.save_weights(config.save_path)
        #         break
        #     except:
        #         print(u'保存失败，正在重试...')


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(corpus(config.data_path), config.batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=config.steps_per_epoch,
        epochs=config.epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights(config.save_path)
