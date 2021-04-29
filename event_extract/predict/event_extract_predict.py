from test4nlp.event_extract.train.event_extract_train import train_model, subject_model, object_model,\
    extract_spoes

while True:
    text = input("请输入文本：")

    result = extract_spoes(text, subject_model, object_model)

    print(result)