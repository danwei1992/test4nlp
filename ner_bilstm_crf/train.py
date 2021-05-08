# import os, sys
# sys.path.append(os.getcwd())
from test4nlp.ner_bilstm_crf import bilsm_crf_model
EPOCHS = 2
model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model()
# train model
model.fit(train_x, train_y,batch_size=16,epochs=EPOCHS, validation_data=[test_x, test_y])
model.save('./resources/ner_bilstm_crf_pretrain/model/crf.h5')
