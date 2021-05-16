from test4nlp.ner_recognition.train.ner_recognition_train import model, NER

text = input('请输入文本：')

result = NER.recognize(text)
print(result)