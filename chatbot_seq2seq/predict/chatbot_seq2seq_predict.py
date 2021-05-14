#! -*- coding: utf-8 -*-
from test4nlp.chatbot_seq2seq.train.chatbot_seq2seq_train import model, tokenizer, chatbot
import json

role_a = input("对话内容：")
content = json.loads(role_a)

print(chatbot.response(content))
