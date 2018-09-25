# -*- coding: utf-8 -*-
from keras.callbacks import Callback
import prepare_data as prd
import configs
import numpy as np
from seqeval.metrics import f1_score
from keras.callbacks import ModelCheckpoint
import model as  mdl

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        pred_label = np.asarray(self.model.predict(self.validation_data[0:2]))
        pred_label = np.argmax(pred_label,axis=-1)
        #Skipping padded sequences
        pred_label = prd.get_orig_labels(pred_label,index_to_label,output_labels_v)
        result  = f1_score(output_labels_v,pred_label)
        print("F1-score--->",result)
        return
 

#Loading training examples/samples
sentences, output_labels, max_length = prd.read_data(configs.TRAINING_FILE)
word_to_index = prd.get_vocabulory(sentences)
label_to_index, index_to_label  = prd.prepare_outputs(output_labels)
char_index = prd.get_vocabulory(word_to_index)
char_indices = prd.get_chars(sentences, max_length, char_index)
vocab_size = len(word_to_index)
glove_vectors = prd.read_glove_vecs(configs.GLOVE_EMBEDDINGS)
word_embeddings = prd.get_preTrained_embeddings(word_to_index,glove_vectors,vocab_size)
max_length = min(configs.MAX_SEQ_LEN, max_length)

with open(configs.DICT_FILE, 'w') as file:
    file.write(str(word_to_index))
    file.write("\n")
    file.write(str(label_to_index))
    file.write("\n")
    file.write(str(max_length))

with open(configs.EMBEDDINGS_FILE, 'wb') as file:
    np.save(file, word_embeddings)

#input and output sequences to the model
train_indeces = prd.get_sequence_indices(sentences, word_to_index, max_length)
labels  = prd.get_sequence_indices(output_labels, label_to_index, max_length)
no_of_classes = len(label_to_index)
no_of_examples = len(sentences)
print('Total no of input sequences:', no_of_examples)
assert (len(train_indeces) == len(labels)),"length of I/O sequences doesn't match"

#validation samples/examples
sentences_v, output_labels_v, max_length_v = prd.read_data(configs.VALIDATION_FILE)
indeces_v = prd.get_sequence_indices(sentences_v, word_to_index, max_length)
labels_v  = prd.get_sequence_indices(output_labels_v, label_to_index, max_length)
char_indices_v = prd.get_chars(sentences_v, max_length, char_index)
assert (len(indeces_v) == len(labels_v)),"length of I/O sequences doesn't match"

model = mdl.get_model(word_embeddings, max_length, len(char_index), no_of_classes)
model.summary()

metrics =  Metrics()
checkpointer = ModelCheckpoint(configs.MODEL_FILE, monitor = 'val_acc', verbose=1, save_best_only=True,save_weights_only=True, period=1, mode='max')

model.fit(x = [train_indeces,char_indices] , y = np.expand_dims(labels,axis=-1), batch_size=configs.BATCH_SIZE,epochs= configs.EPOCHS,
          verbose=1, validation_data=([indeces_v,char_indices_v], np.expand_dims(labels_v,axis=-1)), callbacks = [metrics,checkpointer], shuffle=False)