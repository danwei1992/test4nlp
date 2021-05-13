#文章最大长度
max_p_len = 256
#问题最大长度
max_q_len = 64
#答案最大长度
max_a_len = 32
#问答最大长度
max_qa_len = max_q_len + max_a_len
#每批次大小
batch_size = 8
#批次
epochs = 8
#读取数据路径
data_path = './resources/reading_comprehension_pretrain/data'
#模型保存路径
save_path = './resources/reading_comprehension_pretrain/model'

# bert配置
config_path = './resources/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './resources/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './resources/bert/chinese_L-12_H-768_A-12/vocab.txt'