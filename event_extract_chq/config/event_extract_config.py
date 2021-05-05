#读取数据路径
read_data_path = r'./resources/event_extract_pretrain/data'

#模型保存路径
save_path = './resources/event_extract_chq_pretrain/model'

#基本参数

maxlen = 156
epochs = 10
batch_size = 10
learning_rate = 2e-5

config_path = './resources/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './resources/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './resources/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'




