# -*- coding: utf-8 -*-
import numpy as np
import configs

def read_data(file_path):
    sentences = []
    tokens = []
    tags = []
    output_labels = []
    max_length = 0
    with open(file_path,'r') as file:
        for line in file:
            if(line.split(' ')[0] == "-DOCSTART-" or line == '\n'):
                if(len(tokens)> max_length):
                    max_length = len(tokens)
                tokens = []
                tags = []
                sentences.append(tokens)
                output_labels.append(tags)
            else:
                line_data = line.strip().split(' ')
                tokens.append(line_data[0])
                tags.append(line_data[-1])
           
    sentences.pop(0)
    output_labels.pop(0)          
    return sentences, output_labels, max_length       
                 

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
    with open(glove_file, 'r', encoding = 'utf-8') as file:
        word_to_vec_map = {}
        for line in file:
            line = line.strip().split()
            curr_word = line[0]
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
    return word_to_vec_map     
    

def get_preTrained_embeddings(word_to_index,glove_vectors,vocab_size):
    embed_dim = configs.EMBEDDING_DIM
    embed_matrix = np.zeros((vocab_size, embed_dim))
    for word,i in word_to_index.items():
        if i >= vocab_size:
            continue
        if word in glove_vectors:
            embed_vector =  glove_vectors[word]
            embed_matrix[i] = embed_vector
        else:
            embed_matrix[i] = np.random.normal(embed_dim)
    return embed_matrix
    
     
def prepare_outputs(output_labels):
    unq_labels = set()
    for labels in output_labels:
        for label in labels:
            if label not in unq_labels:
               unq_labels.add(label)
    i=1           
    label_to_index = {}
    index_to_label = {}
    for label in unq_labels:
        label_to_index[label] = i
        index_to_label[i] = label
        i += 1
    return label_to_index, index_to_label            
    

def get_sequence_indices(sentences, word_to_index, max_length):
      no_of_examples  = len(sentences)
      sequences  = np.zeros((no_of_examples, max_length), dtype = np.int32)
      for i in range(no_of_examples):
          words = sentences[i]
          j = 0
          for word in words:
              sequences[i,j] =  word_to_index[word]
              j+=1
      return sequences

if __name__ ==  "__main__":
    sentences, output_labels, max_sent_length = read_data(configs.training_file)
    words_to_index,index_to_words = create_vocabulory(sentences)
    vocab_size = min(len(index_to_words), configs.MAX_NO_OF_WORDS)
    print(get_sequence_indices(sentences,words_to_index,max_sent_length)[1])