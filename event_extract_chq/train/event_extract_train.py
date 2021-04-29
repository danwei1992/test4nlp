from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.backend import K, batch_gather, keras
from bert4keras.layers import LayerNormalization
from keras.layers import *
from keras.models import Model
# import os, sys
# sys.path.append(os.getcwd())
from test4nlp.event_extract_chq.config import event_extract_config as Config
from test4nlp.event_extract_chq.train.utils.data_utils import get_data, data_generator
from tqdm import tqdm
import json
import numpy as np

train_data, valid_data = get_data(Config.read_data_path)
# 建立分词器
tokenizer = Tokenizer(Config.dict_path, do_lower_case=True)


def extrac_trigger(inputs):
    "根据subject_ids从output中取出subject的向量表征"
    output, trigger_ids = inputs
    trigger_ids = K.cast(trigger_ids, 'int32')
    start = batch_gather(output, trigger_ids[:, :1])
    end = batch_gather(output, trigger_ids[:, 1:])
    trigger = K.concatenate([start, end], 2)
    return trigger[:, 0]


def build_model():
    bert_model = build_transformer_model(
        config_path=Config.config_path,
        checkpoint_path=Config.checkpoint_path,
        return_keras_model=False
    )

    # 补充输入
    trigger_labels = Input(shape=(None, 2))
    trigger_ids = Input(shape=(2,))
    subject_labels = Input(shape=(None, 2))
    object_labels = Input(shape=(None, 2))

    # 预测trigger
    output = Dense(units=2, activation='sigmoid', kernel_initializer=bert_model.initializer)(bert_model.model.output)

    trigger_preds = Lambda(lambda x: x ** 2)(output)

    trigger_model = Model(bert_model.inputs, trigger_preds)
    # 传入trigger，预测subject
    output = bert_model.model.layers[-2].get_output_at(-1)
    trigger = Lambda(extrac_trigger)([output, trigger_ids])
    output = LayerNormalization(conditional=True)([output, trigger])
    subject_output = Dense(
        units=2,
        activation='sigmoid',
        kernel_initializer=bert_model.initializer
    )(output)
    subject_preds = Lambda(lambda x: x ** 2)(subject_output)

    subject_model = Model(bert_model.model.inputs + [trigger_ids], subject_preds)

    # 传入trigger，预测object
    object_output = Dense(
        units=2,
        activation='sigmoid',
        kernel_initializer=bert_model.initializer
    )(output)
    object_preds = Lambda(lambda x: x ** 2)(object_output)

    object_model = Model(bert_model.model.input + [trigger_ids], object_preds)


    train_model = Model(
        bert_model.model.inputs + [trigger_labels, trigger_ids, subject_labels, object_labels],
        [trigger_preds, subject_preds, object_preds]
    )

    mask = bert_model.model.get_layer('Embedding-Token').output_mask
    mask = K.cast(mask, K.floatx())

    trigger_loss = K.binary_crossentropy(trigger_labels, trigger_preds)
    trigger_loss = K.mean(trigger_loss, 2)
    trigger_loss = K.sum(trigger_loss * mask) / K.sum(mask)

    subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
    subject_loss = K.mean(subject_loss, 2)
    subject_loss = K.sum(subject_loss * mask) / K.sum(mask)

    object_loss = K.binary_crossentropy(object_labels, object_preds)
    object_loss = K.mean(object_loss, 2)
    object_loss = K.sum(object_loss * mask) / K.sum(mask)

    train_model.add_loss(trigger_loss + subject_loss + object_loss)

    optimizer = Adam(Config.learning_rate)

    train_model.compile(optimizer=optimizer)
    return train_model, trigger_model, subject_model, object_model


def extract_spoes(text, trigger_model, subject_model, object_model):

    tokens = tokenizer.tokenize(text, maxlen=Config.maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=Config.maxlen)

    # 抽取trigger
    trigger_preds = trigger_model.predict([[token_ids], [segment_ids]])
    start = np.where(trigger_preds[0, :, 0] > 0.5)[0]
    end = np.where(trigger_preds[0, :, 1] > 0.4)[0]
    triggers = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            triggers.append((i, j))
    if triggers:
        events = []
        token_ids = np.repeat([token_ids], len(triggers), 0)
        segment_ids = np.repeat([segment_ids], len(triggers), 0)
        triggers = np.array(triggers)

        subject_preds = subject_model.predict([token_ids, segment_ids, triggers])
        object_preds = object_model.predict([token_ids, segment_ids, triggers])

        for trigger, subject_pred, object_pred in zip(triggers, subject_preds, object_preds):
            subject = []
            object = []
            subject_start = np.where(subject_pred[:, 0] > 0.5)[0]
            subject_end = np.where(subject_pred[:, 1] > 0.5)[0]
            object_start = np.where(object_pred[:, 0] > 0.5)[0]
            object_end = np.where(object_pred[:, 1] > 0.5)[0]

            for _start in subject_start:
                for _end in subject_end:
                    if _start <= _end:
                        try:
                            subject.append(
                                (text[mapping[_start][0]:mapping[_end][-1]+1])
                            )
                            break
                        except Exception as err:
                            pass

            for _start in object_start:
                for _end in object_end:
                    if _start <= _end:
                        try:
                            object.append(
                                (text[mapping[_start][0]:mapping[_end][-1]+1])
                            )
                            break
                        except Exception as err:
                            pass
            events.append({
                'trigger': text[mapping[trigger[0]][0]:mapping[trigger[1]][-1]+1],
                'subject': subject,
                'object': object
            })

        return events
    else:
        return []


def evalute(data, trigger_model, subject_model, object_model):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    _pred = set()
    _real = set()
    f = open(Config.read_data_path + '/dev_pred.json', 'w', encoding='utf-8')
    for d in data:
        R = extract_spoes(d['text'], trigger_model, subject_model, object_model)
        for event in R:
            pre_trigger = event['trigger']
            if event['subject']:
                subject = event['subject'][0]
            else:
                subject = ''
            if event['object']:
                object = event['object'][0]
            else:
                object = ''
            _pred.add((pre_trigger, subject))
            _pred.add((pre_trigger, object))
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

    def __init__(self, train_model, trigger_model, subject_model, object_model):
        self.best_val_f1 = 0.
        self.train_model = train_model
        self.trigger_model = trigger_model
        self.subject_model = subject_model
        self.object_model = object_model

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evalute(valid_data, self.trigger_model, self.subject_model, self.object_model)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.train_model.save_weights(Config.save_path + '/best_model.weights')
        print('f1: %.5f, precision: %.5f, reacll: %.5f, best f1: %.5f\n' % (f1, precision, recall, self.best_val_f1))


if __name__ == "__main__":
    train_model, trigger_model, subject_model, object_model = build_model()
    train_generator = data_generator(tokenizer, Config.maxlen, train_data, Config.batch_size)
    evaluator = Evaluator(train_model, trigger_model, subject_model, object_model)
    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=Config.epochs,
        callbacks=[evaluator]
    )
else:
    train_model, trigger_model, subject_model, object_model = build_model()
    print('----------加载模型--------------')
    train_model.load_weights(Config.save_path + '/best_model.weights')
    print('----------模型加载完成--------------')
