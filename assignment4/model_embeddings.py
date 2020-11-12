#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
"""

import torch.nn as nn

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layers.

        @param embed_size (int): Embedding size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size

        # default values
        self.source = None
        self.target = None

        src_pad_token_idx = vocab.src['<pad>']
        tgt_pad_token_idx = vocab.tgt['<pad>']

        # nn.Embedding  --A simple lookup table that stores embeddings of a fixed dictionary and size.
        # torch.nn.Embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
        #                    max_norm: Optional[float] = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False,
        #                    sparse: bool = False, _weight: Optional[torch.Tensor] = None)
        # Embedding Layer for source language
        self.source =  nn.Embedding(len(vocab.src), self.embed_size, padding_idx=src_pad_token_idx)
        # Embedding Layer for target langauge
        self.target =  nn.Embedding(len(vocab.tgt), self.embed_size, padding_idx=tgt_pad_token_idx) 


