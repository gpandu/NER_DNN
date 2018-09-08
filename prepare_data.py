# -*- coding: utf-8 -*-
import pdb

def prepare_data(file_path):
    sentences = []
    tokens = []
    tags = []
    output_labels = []
    with open(file_path) as file:
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
                

if __name__ ==  "__main__":
    sentences, output_labels = prepare_data("data/test.txt")