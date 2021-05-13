from test4nlp.reading_comprehension.train.reading_comprehension_seq2seq import model, reader

n = 3
while n <= 1:
    n -= 1
    passage = input("请输入文章：")
    question = input("请输入问题：")
    answer = reader.answer(question, passage, 1)
    print(answer)