#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, e_char, filter_size, m_word = 21, kernel_size = 5):
        """initial cnn
        @param filter_size (int): the number of filter,also the e_word
        @param e_char (int): the demention of char
        @param kernel_size (int): the filter's length
        """        
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=e_char, out_channels=filter_size, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(m_word - kernel_size + 1)
        
    def forward(self, reshaped) -> torch.Tensor:
        """
        @param reshaped (torch.Tensor): the char embedding of sentences
        @return conv_out (torch.Tensor):the ouput of cnn
        """
        # (batch_size, e_char, max_word_len) -> (batch_size, e_word, max_word_len - kernel_size + 1)
        conv_out = F.relu(self.conv1d(reshaped))
        # (batch_size, e_word, max_word_len - kernel_size + 1) -> (batch_size, e_word, 1)
        conv_out = self.pool(conv_out)
        
        # (batch_size, e_word, 1) -> (batch_size, e_word)
        return conv_out.squeeze(-1)
        
if __name__ == '__main__':
    cnn = CNN(50, 4)
    input = torch.randn(10, 50, 21)
    assert(cnn(input).shape==(10, 4))
