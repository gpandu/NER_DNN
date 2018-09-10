# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Dense,Input, LSTM, Bidirectional, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
import prepare_data as prd
import configs
import numpy as np


def one_hot_encodings(examples,max_length,classes,label):
    encodings = np.zeros((examples,max_length,classes), dtype = np.int32)
    for i,item in enumerate(label):
        encodings[i,:,:] = to_categorical(item,classes)
    return encodings  
    
#Loading training examples/samples
sentences, output_labels, max_length = prd.read_data(configs.training_file)
word_to_index,index_to_words = prd.get_vocabulory(sentences)
label_to_index, index_to_label  = prd.get_vocabulory(output_labels)
vocab_size = min(len(word_to_index), configs.MAX_NO_OF_WORDS)
glove_vectors = prd.read_glove_vecs(configs.glove_embeddings)
word_embeddings = prd.get_preTrained_embeddings(word_to_index,glove_vectors,vocab_size)

#input and output sequences to the model
train_indeces = prd.get_sequence_indices(sentences, word_to_index, max_length)
labels  = prd.get_sequence_indices(output_labels, label_to_index, max_length)
no_of_classes = len(label_to_index)
no_of_examples = len(sentences)
out_encodings = one_hot_encodings(no_of_examples,max_length,no_of_classes,labels)
assert (len(train_indeces) == len(labels)),"length of I/O sequences doesn't match"


#validation samples/examples
sentences_v, output_labels_v, max_length_v = prd.read_data(configs.validation_file)
indeces_v = prd.get_sequence_indices(sentences_v, word_to_index, max_length)
labels_v  = prd.get_sequence_indices(output_labels_v, label_to_index, max_length)
encodings_v = one_hot_encodings(len(sentences_v),max_length,no_of_classes,labels_v)
assert (len(indeces_v) == len(labels_v)),"length of I/O sequences doesn't match"


#Model    
inputs = Input(shape=(max_length,),dtype='int32',name='input')
word_embed = Embedding(input_dim=word_embeddings.shape[0], output_dim=word_embeddings.shape[1],
                       weights=[word_embeddings], trainable=False)(inputs)
output = Bidirectional(LSTM(max_length, return_sequences=True))(word_embed)
output = TimeDistributed(Dense(no_of_classes, activation='softmax'))(output)
model = Model(inputs=[inputs], outputs=[output])
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics = ['accuracy'])
model.summary()

model.fit(x = train_indeces, y = out_encodings, batch_size=configs.batch_size,epochs= configs.batch_size,
          verbose=1, validation_data=(indeces_v, encodings_v))

