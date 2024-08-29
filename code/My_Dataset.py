import spacy 
import torch 
from torchtext.vocab import GloVe
import sys 
import os
import pandas as pd
from typing import List, Tuple
import csv

# print(os.getcwd())
# sys.exit()

class My_Dataset():

    def __init__(self, path_to_train_dataset_csv: str = 'train.csv'):
        '''
        used for the complete pre-processing part and then to get dataloaders as well after converting to tensor embeddings
        '''

        # ATTRIBUTES

        self.train_csv_path = path_to_train_dataset_csv
        self.nlp = spacy.load('en_core_web_sm') # for tokenization
        self.max_len_sent = 64  # if the length of any sentence is bigger than this we truncate it otherwise we pad it with a padding token
        self.padding_token = '<PAD>'

        # READING THE DATASET

        self.tuple_generator = self.read_csv()


    def read_csv(self, colname_1='sentence', colname_2='gold_label'):

        '''
        this creates a generator object that yield 1 tuple of (text, label) in one iteration
        '''

        print("Started to read the data line by line")

        with open(self.train_csv_path, 'r') as f: 
            reader = csv.DictReader(f)
            for row in reader:
                yield row[colname_1], row[colname_2]

    def pre_process_raw_text(self) -> Tuple[List[List[str]], List[List[str]], List[List[int]], List[int]]:

        print("Started pre-processing the data")

        ''' 
        i have the raw_text in a list, which needs to be pre-processed
        i will be using lemmatization and also remove the punctuation marks, stop words
        also the make the sentence to be of equal length, add the <PAD> token to make the length equal to max_len
        (lemma_list, pos_tags_list, masks_list)
        '''

        self.lemmas_list = [] # this is a list of lists, containing the lemmas for each sent in a list
        self.pos_tags_list = [] # list of lists, containing the pos_tags for each sent in a list
        self.masking_list = [] # list of lists, where each list contains 0 or 1 : 0 if the token is self.padding_token
        self.label_list = [] # list containing the labels for each sentence


        for my_index, my_tuple in enumerate(self.tuple_generator):
            print('*' * 20)
            print('\n\n')
            print(f"my_index = {my_index}, STARTED PRE_PROCESSING\n")
            sent, label = my_tuple
            my_lemmas, my_pos_tags, my_masking = self.pre_process_single_sent(sent)
            self.lemmas_list.append(my_lemmas)
            self.pos_tags_list.append(my_pos_tags)
            self.masking_list.append(my_masking)
            self.label_list.append(int(label)+1)
            print(f"my_index = {my_index}, COMPLETED PRE_PROCESSING\n\n")

        print("Pre-processing complete")
        
        return (self.lemmas_list, self.pos_tags_list, self.masking_list, self.label_list)
        
    def pre_process_single_sent(self, sent: str) -> Tuple[List[str], List[str], List[int]]:
        '''
        given a single sentence returns a list of the lemmas in the list along with their Part of speech tags in a seperate list
        and masking (my_lemmas, my_pos, mask), padding or truncation is also done to make the length of each sentence to be same
        each list is of length self.max_len_sent = 256
        '''

        doc = self.nlp(sent.lower())
        my_lemmas = [token.lemma_ for token in doc if token.is_alpha]
        my_pos = [token.pos_ for token in doc if token.is_alpha]

        len_of_my_lemmas = len(my_lemmas)

        if len_of_my_lemmas >= self.max_len_sent:
            # truncate at maximum length of the sentence
            my_lemmas = my_lemmas[:self.max_len_sent]
            my_pos = my_pos[:self.max_len_sent]
            my_mask = [1 for _ in range(self.max_len_sent)]

        else:
            while len(my_lemmas) < self.max_len_sent:
                my_lemmas.append(self.padding_token)
                my_pos.append(self.padding_token)
            
            t1 = [1 for _ in range(len_of_my_lemmas)]
            t0 = [0 for _ in range(self.max_len_sent - len_of_my_lemmas)]
            my_mask = t1 + t0

        return my_lemmas, my_pos, my_mask

