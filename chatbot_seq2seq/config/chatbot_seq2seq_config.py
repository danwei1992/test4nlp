#基本参数
maxlen = 512
batch_size = 10
steps_per_epoch = 1000
epochs = 200

#数据路径
data_path = './resources/chatbot_seq2seq_pretrain/data/LCCD-large-shuf.json'
#补充词表
use_tokens_path = './resources/chatbot_seq2seq_pretrain/data/user_tokens.csv'
#模型保存
save_path = './resources/chatbot_seq2seq_pretrain/model/chatbot_best_model'

# nezha配置
config_path = './resources/bert/nezha_gpt_dialog/config.json'
checkpoint_path = './resources/bert/nezha_gpt_dialog/model.ckpt'
dict_path = './resources/bert/nezha_gpt_dialog/vocab.txt'
