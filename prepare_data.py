# -*- coding: utf-8 -*-
import pdb
import numpy as np

max_words = 30000

def read_data(file_path):
    sentences = []
    tokens = []
    tags = []
    output_labels = []
    with open(file_path,'r') as file:
        for line in file:
            if(line.split(' ')[0] == "-DOCSTART-" or line == '\n'):
                tokens = []
                tags = []
                sentences.append(tokens)
                output_labels.append(tags)
            else:
                tokens.append(line.split(' ')[0])
                tags.append(line.split(' ')[3])
    sentences.pop(0)
    output_labels.pop(0)            
    return sentences, output_labels         
                 

def create_vocabulory(sentences):
    vocabs = {}
    for tokens in sentences:
        for word in tokens:
            if word not in vocabs:
                vocabs[word] = 1
            else:
                vocabs[word] += 1
    words = list(sorted(vocabs, key=vocabs.__getitem__, reverse=True))
    i = 1
    words_to_index = {}
    index_to_words = {}
    for word in words:
        words_to_index[word] = i
        index_to_words[i] = word
        i += 1
    return words_to_index, index_to_words 


def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as file:
        words = set()
        word_to_vec_map = {}
        for line in file:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
    return word_to_vec_map     
    
def get_preTrained_embeddings():
    return None
     

if __name__ ==  "__main__":
    sentences, output_labels = read_data("data/train.txt")
    _,unq_words  = create_vocabulory(sentences)
    print(min(len(unq_words)+1,max_words))