#基本参数
maxlen = 512
batch_size = 10
steps_per_epoch = 1000
epochs = 100

#数据路径
data_path = './resources/chatbot/data/LCCD-large-shuf.json'
#补充词表
use_tokens_path = './resources/chatbot/data/user_tokens.csv'
#模型保存
save_path = './resources/chatbot/model/chatbot_best_model'

# nezha配置
config_path = './resources/bert/nezha_base/bert_config.json'
checkpoint_path = '/resources/bert/nezha_base/model.ckpt-900000'
dict_path = '/resources/bert/nezha_base/vocab.txt'
