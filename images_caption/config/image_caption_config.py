# 模型配置
maxlen = 64
batch_size = 32
steps_per_epoch = 1000
epochs = 10000

# bert配置
config_path = './resources/uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './resources/uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './resources/uncased_L-12_H-768_A-12/vocab.txt'


train_data = './resources/image_caption_pretrain/data/annotations/captions_train2014.json'
val_data = './resources/image_caption_pretrain/data/annotations/captions_val2014.json'
save_path = './resources/image_caption_pretrain/model/best_model.weights'
save_result = './resources/image_caption_pretrain/data/image_result.txt'