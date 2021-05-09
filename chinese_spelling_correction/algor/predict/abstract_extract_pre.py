# -*- coding:utf-8 -*-
import numpy as np
from bert4keras.models import build_transformer_model
from keras.models import Model
from jdqd.a04.abstract_extract.algor.train.utils.abstract_extract_utils import tokenizer, keep_tokens, CrossEntropy
import jdqd.a04.abstract_extract.config.abstract_extract_config as abs_config
from bert4keras.snippets import AutoRegressiveDecoder

def build_model(save_model_path):
    model = build_transformer_model(
        abs_config.config_path,
        abs_config.checkpoint_path,
        application='unilm',
        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    )

    output = CrossEntropy(2)(model.inputs + model.outputs)

    model = Model(model.inputs, output)
    model.load_weights(f'{save_model_path}/abstract_best_model.weights')
    return model

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        result = self.model.predict([token_ids, segment_ids])
        print(result.shape)
        return self.model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, topk=1):
        max_c_len = abs_config.maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk)  # 基于beam search
        return tokenizer.decode(output_ids)

def abs_pre(text, autotitle):
    abs = autotitle.generate(text)
    return abs

if __name__ == "__main__":
    model = build_model(abs_config.save_model_path)
    autotitle = AutoTitle(model=model, start_id=None, end_id=tokenizer._token_end_id, maxlen=32)
    test_path = abs_config.test_data_path
    test_result_path = abs_config.test_data_save
    f_w = open(test_result_path, 'w', encoding='utf-8')
    f = open(test_path, 'r', encoding='utf-8')
    for content in f.readlines():
        print(content)
        abs = abs_pre(content.replace(' ','').replace('\t',' '), autotitle)
        print(abs)
        break
        # f_w.write('conten_lens:' + str(len(content)) + '\n' + 'content:' + content + '\n' + 'abstract:' + abs + '\n\n')