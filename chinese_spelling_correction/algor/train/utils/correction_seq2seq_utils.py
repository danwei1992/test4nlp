import json
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from bert4keras.layers import Loss
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer, load_vocab
import test4nlp.chinese_spelling_correction.config.correction_seq2seq_config as config

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=config.dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


def get_data(train_data_path, valid_data_path):
    train_data = json.load(open(train_data_path, 'r', encoding='utf-8'))
    valid_data = json.load(open(valid_data_path, 'r', encoding='utf-8'))
    return train_data, valid_data


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, d in self.sample(random):
            wrong = d[0]
            right = d[1]
            token_ids, segment_ids = tokenizer.encode(
                wrong, right, maxlen=config.maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []



class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


