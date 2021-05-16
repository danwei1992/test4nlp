#基本参数
maxlen = 256
epochs = 10
batch_size = 32
bert_layers = 12
learning_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率

#数据路径
data_path = './resources/ner_recognition_pretrain/data'

#保存路径
save_path = './resources/ner_recognition_pretrain/model'

# bert配置
config_path = './resources/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './resources/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './resources/bert/chinese_L-12_H-768_A-12/vocab.txt'
