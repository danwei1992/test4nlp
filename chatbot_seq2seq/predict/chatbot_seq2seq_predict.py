#! -*- coding: utf-8 -*-
from test4nlp.chatbot_seq2seq.train.chatbot_seq2seq_train import model, tokenizer, chatbot
import json

def chatbot_seq2seq(dialogue):
    response = chatbot.response(dialogue)
    return response

if __name__ == "__main__":
    dialogue = input("请输入对话：")
    dialogue = json.loads(dialogue)
    result = chatbot_seq2seq(dialogue)
    print(result)