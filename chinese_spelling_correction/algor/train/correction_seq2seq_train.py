# -*- coding:utf-8 -*-
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from keras.models import Model
import os, sys
sys.path.append(os.getcwd())
from test4nlp.chinese_spelling_correction.algor.train.utils.correction_seq2seq_utils import tokenizer, keep_tokens, \
    data_generator, CrossEntropy, get_data
import test4nlp.chinese_spelling_correction.config.correction_seq2seq_config as config
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder


def build_model():
    model = build_transformer_model(
        config.config_path,
        config.checkpoint_path,
        application='unilm',
        keep_tokens=keep_tokens  # 只保留keep_tokens中的字，精简原字表
    )

    output = CrossEntropy(2)(model.inputs + model.outputs)

    model = Model(model.inputs, output)
    model.compile(optimizer=Adam(1e-5))
    return model


model = build_model()


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, topk=1):
        max_c_len = config.maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=64)


def valid_show(valid_data):
    f = open(f'{config.save_model_path}/result.txt', 'w', encoding='utf-8')
    for v in valid_data:
        f.write('real:' + v[1] + '\n' + 'pred:' + autotitle.generate(v[0]) + '\n\n')


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights(f'{config.save_model_path}/correction_seq2seq_best_model.weights')


if __name__ == "__main__":
    train_data, valid_data = get_data(config.train_data_path, config.valid_data_path)
    train_generator = data_generator(train_data, config.batch_size)

    evaluator = Evaluator()
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=config.steps_per_epoch,
        epochs=config.epochs,
        callbacks=[evaluator]
        )
else:
    print("-------------加载模型-------------")
    model.load_weights(f'{config.save_model_path}/correction_seq2seq_best_model.weights')
    print("-----------模型加载完毕------------")

