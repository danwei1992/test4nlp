# content+abstract最大长度
maxlen = 126
#训练批次
epochs = 15
#每批次大小
batch_size = 12
# 模型参数
config_path = './resources/chinese_L-12_H-768_A-12/bert_config.json'
# 初始化模型路径
checkpoint_path = './resources/chinese_L-12_H-768_A-12/bert_model.ckpt'
# 模型字典路径
dict_path = './resources/chinese_L-12_H-768_A-12/vocab.txt'
# 每打印一次epochs数
steps_per_epoch = 100
# 训练数据路径
train_data_path = './resources/chinese_spelling_correction_pretrain/data/train_data.json'
# 测试数据路径
valid_data_path = './resources/chinese_spelling_correction_pretrain/data/valid_data.json'
# 测试数据保存文件
valid_data_save = './resources/chinese_spelling_correction_pretrain/data'
# 模型保存路径
save_model_path = './resources/chinese_spelling_correction_pretrain/model'
