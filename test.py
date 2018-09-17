# -*- coding: utf-8 -*-
import prepare_data as prd
import configs
import numpy as np
from seqeval.metrics import f1_score
import model_training


def remove_sents(sentences_t,labels_t,max_length):
    remove_idxs = []
    for i,sentence in enumerate(sentences_t):
        if len(sentence)>max_length:
            remove_idxs.append(i)
    for i in remove_idxs:
        sentences_t.pop(i)
        labels_t.pop(i)
    
    
word_index = {}
label_index = {}
max_length = 0
 
with open(configs.dict_file, 'r') as file:
    dicts  = file.read()
    dicts = dicts.split("\n")
    word_index = eval(dicts[0])
    label_index = eval(dicts[1])
    max_length  = eval(dicts[2])
    
#Loading test sequences    
sentences_t, labels_t, max_length_t = prd.read_data(configs.test_file)
remove_sents(sentences_t, labels_t, max_length)
char_index = prd.get_vocabulory(word_index)
char_idxs = prd.get_chars(sentences_t, max_length, char_index)
label_idxs  = prd.get_sequence_indices(labels_t, label_index, max_length)
seq_idxs = prd.get_sequence_indices(sentences_t, word_index, max_length)

index_labels = {}
for item,i in label_index.items():
    index_labels[i] = item 

model = model_training.get_model()
model.load_weights(configs.model_file)   

# Predict labels for test data
pred_label = np.asarray(model.predict([seq_idxs,char_idxs]))
pred_label = np.argmax(pred_label,axis=-1)
#Skipping padded sequences
pred_label = prd.get_orig_labels(pred_label,index_labels,labels_t)
result  = f1_score(labels_t,pred_label)
print("F1-score--->",result)
print("complete")