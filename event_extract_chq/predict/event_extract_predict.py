from test4nlp.event_extract_chq.train.event_extract_train import train_model, trigger_model, subject_model, \
    object_model,extract_spoes

while True:
    text = input("请输入文本：")

    result = extract_spoes(text, trigger_model, subject_model, object_model)

    print(result)