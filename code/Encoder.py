import torch 
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import List
from utils import POS_VOCAB

class Encoder_Sentiment_Analysis(nn.Module):

    def __init__(self, input_dim: int, num_heads: int, pos_tags_vocab_len: int = len(POS_VOCAB), output_dim: int = 3, dim_feedfwd: int = 2048, dropout_prob: int=0.1, activation: str='relu', num_layers: int=4):

        '''
        first will be an Transformer encoding layer, then will use some method of pooling to convert 3D tensors to 2D tensors and
        then passing through feed fwd network to get the logit scores and then using cross entropy loss function

        '''

        super().__init__()

        # self.dataset = dataset
        self.num_heads = num_heads
        self.dim_feedfwd = dim_feedfwd
        self.input_dim = input_dim
        self.dropout_prob = dropout_prob
        self.activation = activation
        self.num_layers = num_layers
        self.pos_tags_vocab_len = pos_tags_vocab_len
        self.output_dim = output_dim

        self.encoder_layer = TransformerEncoderLayer(d_model=self.input_dim, nhead=self.num_heads, dim_feedforward=self.dim_feedfwd,
                                                     dropout=self.dropout_prob, activation=self.activation, batch_first=True)
        self.encoder = TransformerEncoder(self.encoder_layer, self.num_layers)

        self.feed_fwd_part_end = nn.Sequential(
            nn.Linear(self.input_dim * 3, self.input_dim),
            nn.ReLU(), nn.Dropout(p=0.3),
            nn.Linear(self.input_dim, self.output_dim)
        )

        self.pos_embedding = nn.Embedding(num_embeddings=self.pos_tags_vocab_len, embedding_dim=self.input_dim, padding_idx=POS_VOCAB['<PAD>']) # to be learned

    def forward(self, inputs):

        ''''
        given a batch of inputs, give the outputs

        input : will be a list of two elements
        input_list : (lemma_embed+positional_embeddings) [Tensor], list of pos-tags mapped to their indices 
        first tensor is lemma_emebeddings + positonal_embeddings : shape = (batch_size, seq_len=256, input_dim=50)
        second tensor is pos_indices of each sentence : shape = (batch_size, seq_len)
        '''

        pos_tags_embeds = self.pos_embedding(inputs[1])
        features = pos_tags_embeds + inputs[0]

        out = self.encoder(features)
        # need to apply some pooling here 

        max_pool, _ = torch.max(out, dim=1)
        min_pool, _ = torch.min(out, dim=1)
        avg_pool = torch.mean(out, dim=1)

        out = torch.cat([max_pool, min_pool, avg_pool], dim=1) # dim = (batch_size, 3*embed_dim)

        # now shape is (batch_size, 3 * embed_dim)
        # then pass through feed forward neural network to get some raw scores before feeding into the loss function

        ans = self.feed_fwd_part_end(out)
        return ans






        
