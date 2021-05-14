#! -*- coding: utf-8 -*-
from test4nlp.chatbot_seq2seq.train.chatbot_seq2seq_train import model, tokenizer, chatbot

def chatbot_seq2seq(dialogue):
    response = chatbot.response(dialogue)
    return response

