# -*- coding: utf-8 -*-

import time
import json
import numpy as np
from test4nlp.multi_label_classific.train.multi_label_classific_train import model, label_dict, predict_single_text
import test4nlp.multi_label_classific.config.multi_label_classific_config as config

content = input("请输入文本：")

result = predict_single_text(content)

print(result)
