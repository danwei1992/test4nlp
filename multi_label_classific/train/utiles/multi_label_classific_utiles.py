import pandas as pd
import numpy as np
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
import test4nlp.multi_label_classific.config.multi_label_classific_config as config
import os, json

# 建立分词器
tokenizer = Tokenizer(config.dict_path, do_lower_case=True)

def getdata(data_path):
    train_df = pd.read_csv(os.path.join(config.data_path, "train.csv")).fillna(value="")
    test_df = pd.read_csv(os.path.join(config.data_path, "test.csv")).fillna(value="")
    select_labels = train_df["label"].unique()
    labels = []
    for label in select_labels:
        if "|" not in label:
            if label not in labels:
                labels.append(label)
        else:
            for _ in label.split("|"):
                if _ not in labels:
                    labels.append(_)
    label_dict = dict(zip(range(len(labels)), labels))
    with open(os.path.join(config.save_path, "label.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(label_dict, ensure_ascii=False, indent=2))

    train_data = []
    for i in range(train_df.shape[0]):
        label, content = train_df.iloc[i, :]
        label_id = [0] * len(labels)
        for j, _ in enumerate(labels):
            for separate_label in label.split("|"):
                if _ == separate_label:
                    label_id[j] = 1
        train_data.append((content, label_id))

    return train_data, test_df, label_dict


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        X1, X2, Y = [], [], []
        for is_end, d in self.sample(random):
            text = d[0][:config.maxlen]
            x1, x2 = tokenizer.encode(text)
            y = d[1]
            X1.append(x1)
            X2.append(x2)
            Y.append(y)
            if len(X1) == self.batch_size or is_end:
                X1 = sequence_padding(X1)
                X2 = sequence_padding(X2)
                Y = sequence_padding(Y)
                yield [X1, X2], Y
                [X1, X2, Y] = [], [], []