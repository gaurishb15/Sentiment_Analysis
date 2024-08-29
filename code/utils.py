import torch 
import torch.nn as nn
from My_Dataset import My_Dataset
from typing import List
from torchtext.vocab import GloVe
from torch.utils.data import Dataset
import math

# GLOBAL VARS:

POS_VOCAB = {'ADJ': 0, 'ADP': 1, 'ADV': 2, 'AUX': 3, 'CONJ': 4, 'CCONJ': 5, 'DET': 6,
                          'INTJ': 7, 'NOUN': 8, 'NUM': 9, 'PART': 10, 'PRON': 11, 'PROPN': 12,
                          'PUNCT': 13, 'SCONJ': 14, 'SYM': 15, 'VERB': 16, '<UNK>': 17, '<PAD>': 18}

# CLASSES:

class JustDataset(Dataset):

    def __init__(self, data):
        self.data = data 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

# FUNCTIONS:

def convert_pos_tags_list_into_list_of_indices_given_vocab(pos_tags_list: List[List[str]], pos_tag_vocab: dict = POS_VOCAB) -> List[List[int]]:

        '''
        given the pos_tags for all sentences converts them into a list of tensors
        each tensor is of shape : (seq_len, )
        if it is a padding token then i give the embedding to be 0
        '''

        print("Started to map pos_tags into indices using vocab")

        pos_tags_list_std = [keys for keys, values in pos_tag_vocab.items()] # contains all the pos tags we are using

        all_indices_list = []
        for pos_sent in pos_tags_list:
            pos_index_list = []
            for pos in pos_sent:
                pos_index = pos_tag_vocab[pos] if pos in pos_tags_list_std else pos_tag_vocab['<UNK>']
                pos_index_list.append(pos_index)
            all_indices_list.append(pos_index_list)

        print("Done mapping pos_tags to index ")
        print(len(all_indices_list))

        return all_indices_list

def get_positional_encodings(masking_list: List[List[int]], sent_len: int = 64, input_dim: int=100) -> List[torch.Tensor]:
        
        '''
        given a list of lists, where each sublist contains mask (0 or 1) values, 0 correspond to padding tokens
        return the positional encoding for every sentence in the form of a list of tensors
        '''

        ans_list = []

        for my_index, mask in enumerate(masking_list):
            print('*' * 100)
            print('\n\n')
            print(f"my_index = {my_index}")
            pos_encoding = get_positional_encoding_single_sent(mask, sent_len, input_dim)
            print("got the positional_encoding")
            ans_list.append(pos_encoding)

        return ans_list
        # pos_encodings = torch.stack(ans_list)

def get_positional_encoding_single_sent(mask: List[int], sent_len: int = 64, input_dim: int = 100) -> torch.Tensor:

    # Create a tensor with shape (sent_len, input_dim)

    print("Start generating positional encodings")

    position = torch.arange(0, sent_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, input_dim, 2, dtype=torch.float) * -(math.log(10000.0) / input_dim))
    encodings = torch.zeros((sent_len, input_dim))
    encodings[:, 0::2] = torch.sin(position * div_term)
    encodings[:, 1::2] = torch.cos(position * div_term)
    # Apply the mask to zero out the padded tokens
    encodings = encodings.masked_fill(torch.LongTensor(mask).unsqueeze(1) == 0, 0)

    print("Got the positional encoding tensors")

    return encodings

def convert_lemmas_list_into_tensor_embeddings(lemmas_list: List[List[str]], input_dim: int = 100) -> List[torch.Tensor]:

    '''
     given the lemmas_list from pre-processing convert it into tensors embeddings using pre-trained glove
    '''
    
    assert input_dim in [50, 100, 200, 300]

    print("Started converting the lemmas into tensor embeddings")

    glove = GloVe(name='6B', dim=input_dim, cache='glove.6B') # to convert text into embeddings
    lemmas_embeddings_list = []
    for sent in lemmas_list:
        temp = []
        for my_lemma in sent: 
            my_tensor = glove[my_lemma]
            temp.append(my_tensor)
        temp = torch.stack(temp)
        lemmas_embeddings_list.append(temp)

    print("Converted all the lemmas into embeddings")

    return lemmas_embeddings_list

def prepare_data(my_dataset: My_Dataset, input_dim: int = 100):

    lemmas_list, pos_tags_list, masking_list, labels_list = my_dataset.pre_process_raw_text() 
    labels_list = list(map(int, labels_list)) 

    lemma_tensors = convert_lemmas_list_into_tensor_embeddings(lemmas_list, input_dim)
    positional_tensors = get_positional_encodings(masking_list, my_dataset.max_len_sent, input_dim)
    pos_indices = convert_pos_tags_list_into_list_of_indices_given_vocab(pos_tags_list)

    lemma_combine_positional = []
    for x, y in zip(lemma_tensors, positional_tensors):
        temp = x + y
        lemma_combine_positional.append(temp)

    data = []
    for i in range(len(lemmas_list)):
        temp = (lemma_combine_positional[i], pos_indices[i], labels_list[i])
        data.append(temp)

    torch.save(data, 'my_dataset_max_len_64_input_dim_100.pt')
    print("Successfully saved the dataset")
    
