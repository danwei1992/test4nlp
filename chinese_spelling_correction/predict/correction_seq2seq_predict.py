import os, sys
sys.path.append(os.getcwd())
from test4nlp.chinese_spelling_correction.train.correction_seq2seq import model, get_answer

while True:
    text = input("请输入文本：")
    answer = get_answer(text)
    print(answer)
