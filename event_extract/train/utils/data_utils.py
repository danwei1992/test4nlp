from test4nlp.event_extract.config import event_extract_config as Config
import pandas as pd
import numpy as np
from bert4keras.snippets import DataGenerator, sequence_padding


def get_data(path):
    # 读取数据
    train = pd.read_csv(path + '/chusai_data/train/train1.csv', encoding='utf-8')
    train.fillna('', inplace=True)
    # test = pd.read_csv(path+'/chusai_data/test/test1.csv', encoding='utf-8')

    # 谓语动词编码
    triger_lists = list(set(train['trigger'].tolist()))
    predicate2id = {j: i for i, j in enumerate(triger_lists)}
    id2predicate = {i: j for i, j in enumerate(triger_lists)}

    # 生成训练集与验证集
    train_dict = {}
    for k, v1, v2, v3 in zip(train['news'], train['trigger'], train['object'], train['subject']):
        if k not in train_dict:
            train_dict[k] = []
            train_dict[k].append({'trigger': v1, 'object': v2, 'subject': v3})
        else:
            train_dict[k].append({'trigger': v1, 'object': v2, 'subject': v3})

    train_data_all = []
    for k, v in train_dict.items():
        train_data_all.append({'text': k, 'events': v})

    train_data = train_data_all[:6000]
    valid_data = train_data_all[6000:]

    return predicate2id, id2predicate, train_data, valid_data


def search(pattern, sequence):
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(DataGenerator):
    def __init__(self, tokenizer, predicate2id, maxlen, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.predicate2id = predicate2id
        self.maxlen = maxlen

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
        for is_end, d in self.sample(random):
            token_ids, segment_ids = self.tokenizer.encode(d['text'], maxlen=self.maxlen)
            # 整理三元组
            spoes = {}
            for event in d['events']:
                p = event['trigger']
                s = event['subject']
                o = event['object']
                p = self.predicate2id[p]
                s = self.tokenizer.encode(s)[0][1:-1]
                o = self.tokenizer.encode(o)[0][1:-1]
                s_idx = search(s, token_ids)
                o_idx = search(o, token_ids)
                if s_idx != -1 and o_idx != -1:
                    s = (s_idx, s_idx + len(s) - 1)
                    o = (o_idx, o_idx + len(o) - 1, p)
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)

            if spoes:
                # subject标签
                subject_labels = np.zeros((len(token_ids), 2))
                for s in spoes:
                    subject_labels[s[0], 0] = 1
                    subject_labels[s[1], 1] = 1
                # 随机选一个subject
                start, end = np.array(list(spoes.keys())).T
                start = np.random.choice(start)
                end = np.random.choice(end[end >= start])
                subject_ids = (start, end)

                # 对应的object的标签
                object_labels = np.zeros((len(token_ids), len(self.predicate2id), 2))
                for o in spoes.get(subject_ids, []):
                    object_labels[o[0], o[2], 0] = 1
                    object_labels[o[1], o[2], 1] = 1

                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_subject_labels.append(subject_labels)
                batch_subject_ids.append(subject_ids)
                batch_object_labels.append(object_labels)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_subject_labels = sequence_padding(batch_subject_labels, padding=np.zeros(2))
                    batch_subject_ids = np.array(batch_subject_ids)
                    batch_object_labels = sequence_padding(batch_subject_labels,
                                                           padding=np.zeros((len(self.predicate2id), 2)))
                    yield [
                              batch_token_ids, batch_segment_ids,
                              batch_subject_labels, batch_subject_ids,
                              batch_object_labels
                          ], None
                    batch_token_ids, batch_segment_ids = [], []
                    batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
