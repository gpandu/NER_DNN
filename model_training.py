# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Dense,Input, LSTM, Bidirectional, TimeDistributed, Dropout, concatenate, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.callbacks import Callback
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
import prepare_data as prd
import configs
import numpy as np
from seqeval.metrics import f1_score
from keras.callbacks import ModelCheckpoint

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        pred_label = np.asarray(self.model.predict(self.validation_data[0:2]))
        pred_label = np.argmax(pred_label,axis=-1)
        #Skipping padded sequences
        y_pred = prd.get_orig_labels(pred_label,index_to_label,output_labels_v)
        y_true = output_labels_v
        result  = f1_score(y_true,y_pred)
        print("F1-score--->",result)
        return
 

def one_hot_encodings(examples,max_length,classes,label):
    encodings = np.zeros((examples,max_length,classes), dtype = np.int32)
    for i,item in enumerate(label):
        encodings[i,:,:] = to_categorical(item,classes)
    return encodings  
    
#Loading training examples/samples
sentences, output_labels, max_length = prd.read_data(configs.training_file)
word_to_index = prd.get_vocabulory(sentences)
label_to_index, index_to_label  = prd.prepare_outputs(output_labels)
char_index = prd.get_vocabulory(word_to_index)
char_indices = prd.get_chars(sentences, max_length, char_index)
vocab_size = min(len(word_to_index), configs.MAX_NO_OF_WORDS)
glove_vectors = prd.read_glove_vecs(configs.glove_embeddings)
word_embeddings = prd.get_preTrained_embeddings(word_to_index,glove_vectors,vocab_size)

#input and output sequences to the model
train_indeces = prd.get_sequence_indices(sentences, word_to_index, max_length)
labels  = prd.get_sequence_indices(output_labels, label_to_index, max_length)
no_of_classes = len(label_to_index)
no_of_examples = len(sentences)
assert (len(train_indeces) == len(labels)),"length of I/O sequences doesn't match"


#validation samples/examples
sentences_v, output_labels_v, max_length_v = prd.read_data(configs.validation_file)
indeces_v = prd.get_sequence_indices(sentences_v, word_to_index, max_length)
labels_v  = prd.get_sequence_indices(output_labels_v, label_to_index, max_length)
char_indices_v = prd.get_chars(sentences_v, max_length, char_index)
assert (len(indeces_v) == len(labels_v)),"length of I/O sequences doesn't match"


#Model 
chars_input = Input(shape=(max_length,configs.max_chars,), dtype='int32', name='char_input')
chars_emb = TimeDistributed(Embedding(input_dim = len(char_index), output_dim = configs.char_embds, trainable=True, name='char_emb'))(chars_input)
chars_cnn = TimeDistributed(Conv1D(kernel_size=3, filters=configs.no_filters, padding='same',activation='tanh', strides=1))(chars_emb) 
max_out = TimeDistributed(MaxPooling1D(pool_size=configs.pool_size))(chars_cnn) 
chars = TimeDistributed(Flatten())(max_out)
chars = Dropout(0.5)(chars)

words_input = Input(shape=(max_length,),dtype='int32',name='word_input')
word_embed = Embedding(input_dim=word_embeddings.shape[0], output_dim=word_embeddings.shape[1],
                       weights=[word_embeddings], trainable=False, name='word_embed')(words_input)

output = concatenate([word_embed,chars])
output = Bidirectional(LSTM(max_length, return_sequences=True))(output)
output = TimeDistributed(Dense(no_of_classes, activation='softmax'))(output)
model = Model(inputs=[words_input, chars_input], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics = ['accuracy'])
model.summary()

metrics =  Metrics()
checkpointer = ModelCheckpoint(configs.weights_file, verbose=1, save_best_only=True, period=10)
model.fit(x = [train_indeces,char_indices] , y = np.expand_dims(labels,axis=-1), batch_size=configs.batch_size,epochs= configs.epochs,
          verbose=1, validation_data=([indeces_v,char_indices_v], np.expand_dims(labels_v,axis=-1)), callbacks = [metrics,checkpointer])