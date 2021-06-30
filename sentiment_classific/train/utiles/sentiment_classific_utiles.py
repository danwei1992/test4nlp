import os
import test4nlp.sentiment_classific.config.sentiment_classific_config as config



def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    categories2id = {}
    id2categories = {}
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    for k ,v in zip(categories, range(len(categories))):
        categories2id[k] = v
        id2categories[v] = k
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            label, text = l.strip().split('\t')
            D.append((text, categories2id[label]))
    return D

def getdata(data_path):
    # 加载数据集
    train_data = load_data(os.path.join(config.data_path, 'cnews.train.txt'))
    valid_data = load_data(os.path.join(config.data_path, 'cnews.val.txt'))
    test_data = load_data(os.path.join(config.data_path, 'cnews.test.txt'))
    return train_data, valid_data, test_data
