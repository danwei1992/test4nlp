# -*- coding: utf-8 -*-
import os, sys
import numpy as np
# sys.path.append(os.getcwd())
from bert4keras.backend import keras, set_gelu
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from keras.layers import Lambda, Dense
import test4nlp.sentiment_classific.config.sentiment_classific_config as config
from test4nlp.multi_label_classific.train.utiles.multi_label_classific_utiles import tokenizer, getdata, data_generator

set_gelu('tanh')  # 切换gelu版本
train_data, test_df, label_dict = getdata(config.data_path)
print(len(train_data))
print(len(label_dict))

# 加载预训练模型
bert = build_transformer_model(
    config_path=config.config_path,
    checkpoint_path=config.checkpoint_path,
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
output = Dense(
    units=len(label_dict),
    activation='sigmoid',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5),
    metrics=['accuracy'],
)

# 对单句话进行预测
def predict_single_text(text):
    # 利用BERT进行tokenize
    text = text[:config.maxlen]
    x1, x2 = tokenizer.encode(text)
    X1 = x1 + [0] * (config.maxlen - len(x1)) if len(x1) < config.maxlen else x1
    X2 = x2 + [0] * (config.maxlen - len(x2)) if len(x2) < config.maxlen else x2

    # 模型预测并输出预测结果
    prediction = model.predict([[X1], [X2]])
    one_hot = np.where(prediction > 0.5, 1, 0)[0]
    return "|".join([label_dict[str(i)] for i in range(len(one_hot)) if one_hot[i]])


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_precious = 0.

    def on_epoch_end(self, epoch, logs=None):
        precious = evaluate()
        if precious > self.best_precious:
            self.best_precious = precious
            model.save_weights(os.path.join(config.save_path, 'best_model.weights'))
        print(
            u'precious: %.5f, self.best_precious: %.5f\n' %
            (precious, self.best_precious)
        )

# 模型评估
def evaluate():
    common_cnt = 0
    for i in range(test_df.shape[0]):
        true_label, content = test_df.iloc[i, :]
        true_y = [0] * len(label_dict.keys())
        for key, value in label_dict.items():
            if value in true_label:
                true_y[int(key)] = 1

        pred_label = predict_single_text(content)
        if set(true_label.split("|")) == set(pred_label.split("|")):
            common_cnt += 1
        precious = common_cnt / len(test_df)
    return precious


if __name__ == '__main__':
    # 模型训练
    train_D = data_generator(train_data, config.batch_size)

    evaluator = Evaluator()

    model.fit(
        train_D.forfit(),
        steps_per_epoch=len(train_D),
        epochs=10,
        callbacks=[evaluator]
    )

else:
    print('-------------加载模型----------------')
    model.load_weights(os.path.join(config.save_path, 'best_model.weights'))
    print('-------------模型加载成功-------------')