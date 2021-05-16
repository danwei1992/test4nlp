import os
import test4nlp.sentiment_classific.config.sentiment_classific_config as config



def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D

def getdata(data_path):
    # 加载数据集
    train_data = load_data(os.path.join(config.data_path, 'sentiment.train.data'))
    valid_data = load_data(os.path.join(config.data_path, 'sentiment.valid.data'))
    test_data = load_data(os.path.join(config.data_path, 'sentiment.test.data'))
    return train_data, valid_data, test_data
