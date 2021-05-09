# -*- coding:utf-8 -*-
from test4nlp.chinese_spelling_correction.algor.train.correction_seq2seq_train import autotitle, model

text = input("请输入文本: ")
result = autotitle.generate(text)
print(result)
