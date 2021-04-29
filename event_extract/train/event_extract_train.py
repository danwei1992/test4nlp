from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array
from bert4keras.optimizers import Adam
from bert4keras.backend import K, batch_gather, keras
from bert4keras.layers import LayerNormalization
from keras.layers import *
from keras.models import Model
# import os, sys
# sys.path.append(os.getcwd())
from test4nlp.event_extract.config import event_extract_config as Config
from test4nlp.event_extract.train.utils.data_utils import get_data, data_generator
from tqdm import tqdm
import json
import numpy as np

predicate2id, id2predicate, train_data, valid_data = get_data(Config.read_data_path)
# 建立分词器
tokenizer = Tokenizer(Config.dict_path, do_lower_case=True)


def extrac_subject(inputs):
    "根据subject_ids从output中取出subject的向量表征"
    output, subject_ids = inputs
    subject_ids = K.cast(subject_ids, 'int32')
    start = batch_gather(output, subject_ids[:, :1])
    end = batch_gather(output, subject_ids[:, 1:])
    subject = K.concatenate([start, end], 2)
    return subject[:, 0]


def build_model():
    bert_model = build_transformer_model(
        config_path=Config.config_path,
        checkpoint_path=Config.checkpoint_path,
        return_keras_model=False
    )

    # 补充输入
    subject_labels = Input(shape=(None, 2))
    subject_ids = Input(shape=(2,))
    object_labels = Input(shape=(None, len(predicate2id), 2))

    # 预测subject
    output = Dense(units=2, activation='sigmoid', kernel_initializer=bert_model.initializer)(bert_model.model.output)

    subject_preds = Lambda(lambda x: x ** 2)(output)

    subject_model = Model(bert_model.inputs, subject_preds)
    # 传入subject，预测object
    output = bert_model.model.layers[-2].get_output_at(-1)
    subject = Lambda(extrac_subject)([output, subject_ids])
    output = LayerNormalization(conditional=True)([output, subject])
    output = Dense(
        units=len(predicate2id) * 2,
        activation='sigmoid',
        kernel_initializer=bert_model.initializer
    )(output)
    output = Lambda(lambda x: x ** 4)(output)
    object_preds = Reshape((-1, len(predicate2id), 2))(output)

    object_model = Model(bert_model.model.inputs + [subject_ids], object_preds)

    # 训练模型
    train_model = Model(
        bert_model.model.inputs + [subject_labels, subject_ids, object_labels],
        [subject_preds, object_preds]
    )

    mask = bert_model.model.get_layer('Embedding-Token').output_mask
    mask = K.cast(mask, K.floatx())

    subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
    subject_loss = K.mean(subject_loss, 2)
    subject_loss = K.sum(subject_loss * mask) / K.sum(mask)

    object_loss = K.binary_crossentropy(object_labels, object_preds)
    object_loss = K.sum(K.mean(object_loss, 3), 2)
    object_loss = K.sum(object_loss * mask) / K.sum(mask)

    train_model.add_loss(subject_loss + object_loss)

    optimizer = Adam(Config.learning_rate)

    train_model.compile(optimizer=optimizer)
    return train_model, subject_model, object_model


def extract_spoes(text, subject_model, object_model):

    tokens = tokenizer.tokenize(text, maxlen=Config.maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=Config.maxlen)

    # 抽取subject
    subject_preds = subject_model.predict([[token_ids], [segment_ids]])
    start = np.where(subject_preds[0, :, 0] > 0.5)[0]
    end = np.where(subject_preds[0, :, 1] > 0.4)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))
    print(subjects)
    if subjects:
        spons = []
        token_ids = np.repeat([token_ids], len(subjects), 0)
        segment_ids = np.repeat([segment_ids], len(subjects), 0)
        subjects = np.array(subjects)
        object_preds = object_model.predict([token_ids, segment_ids, subjects])
        for subject, object_pred in zip(subjects, object_preds):
            start = np.where(object_pred[:, :, 0] > 0.5)
            end = np.where(object_pred[:, :, 1] > 0.5)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    try:
                        if _start <= _end and predicate1 == predicate2:
                            spons.append(
                                ((mapping[subject[0]][0], mapping[subject[1]][-1]), predicate1,
                                 (mapping[_start][0], mapping[_end][-1]))
                            )
                            break
                    except Exception as err:
                        pass
        return [(text[s[0]:s[1] + 1], id2predicate[p], text[o[0]:o[1] + 1]) for s, p, o in spons]
    else:
        return []


def evalute(data, subject_model, object_model):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    _pred = set()
    _real = set()
    f = open(Config.read_data_path + '/dev_pred.json', 'w', encoding='utf-8')
    for d in data:
        R = extract_spoes(d['text'], subject_model, object_model)
        for event in R:
            _pred.add((event[1], event[0]))
            _pred.add((event[1], event[2]))
        for event in d['events']:
            _real.add((event['trigger'], event['subject']))
            _real.add((event['trigger'], event['object']))


        X += len(_pred & _real)
        Y += len(_pred)
        Z += len(_real)


        s = json.dumps({
            'text': d['text'],
            'events': list(_real),
            'events_pre': list(_pred),
        },
            ensure_ascii=False,
            indent=4
        )
        f.write(s + '\n')
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    f.close()
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    "评估并保存模型"

    def __init__(self, train_model, subject_model, object_model):
        self.best_val_f1 = 0.
        self.train_model = train_model
        self.subject_model = subject_model
        self.object_model = object_model

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evalute(valid_data, self.subject_model, self.object_model)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.train_model.save_weights(Config.save_path + '/best_model.weights')
        print('f1: %.5f, precision: %.5f, reacll: %.5f, best f1: %.5f\n' % (f1, precision, recall, self.best_val_f1))


if __name__ == "__main__":
    train_model, subject_model, object_model = build_model()
    train_generator = data_generator(tokenizer, predicate2id, Config.maxlen, train_data, Config.batch_size)
    evaluator = Evaluator(train_model, subject_model, object_model)
    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=Config.epochs,
        callbacks=[evaluator]
    )
else:
    train_model, subject_model, object_model = build_model()
    print('----------加载模型--------------')
    train_model.load_weights(Config.save_path + '/best_model.weights')
    print('----------模型加载完成--------------')
