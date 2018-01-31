from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata, string, re, random, time, datetime, math
import os, codecs

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy
import numpy as np

import torchtext
from collections import Counter

class Lang:
    def __init__(self, name):
        self.name = name
        self.vocab = None

    def _read_vocab(self, filename, data_path):
        vocabfile = os.path.join(data_path, filename)
        vocab = codecs.open(vocabfile, encoding='utf-8').read().strip().split('\n')

        vocab_counter = Counter()
        for word in vocab:
          vocab_counter[word] += 1

        return vocab_counter

    def make_vocab(self, filename, data_path):
        vocab_counter = self._read_vocab(filename, data_path)
        specials = ['<pad>', '<sos>', '<eos>']
        self.vocab = torchtext.vocab.Vocab(vocab_counter, specials=specials)


def read_parallel_data(data_path, source_filename, target_filename):
    # Read the file and split into lines
    source_path = os.path.join(data_path, source_filename)
    target_path = os.path.join(data_path, target_filename)
    
    source_seqs = codecs.open(source_path, encoding='utf-8').read().strip().split('\n')
    target_seqs = codecs.open(target_path, encoding='utf-8').read().strip().split('\n')
    
    # Normalize sentences, make into pairs and sort in descending order
    source_seqs = [line.lower().strip() for line in source_seqs]
    target_seqs = [line.lower().strip() for line in target_seqs]
    
    # sorted_pairs = sorted(zip(source_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)                             

    return zip(source_seqs, target_seqs)

def filter_pairs_length(pairs, MIN_LENGTH, MAX_LENGTH):
    filtered_pairs = []
    for pair in pairs:
        if len(pair[0].split()) >= MIN_LENGTH and len(pair[0].split()) <= MAX_LENGTH \
            and len(pair[1].split()) >= MIN_LENGTH and len(pair[1].split()) <= MAX_LENGTH:
                filtered_pairs.append(pair)
    return filtered_pairs


def filter_oov_words(pairs, input_lang, output_lang):

    keep_pairs = []

    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        new_input = []
        new_output = []

        for word in input_sentence.split(' '):
            if word not in input_lang.vocab.stoi:
                new_input.append(input_lang.vocab.stoi['<unk>'])
            else:
                new_input.append(input_lang.vocab.stoi[word])
        new_input.append(input_lang.vocab.stoi['<eos>'])

        for word in output_sentence.split(' '):
            if word not in output_lang.vocab.stoi:
                new_output.append(output_lang.vocab.stoi['<unk>'])
            else:
                new_output.append(output_lang.vocab.stoi[word])
        new_input.append(output_lang.vocab.stoi['<eos>'])
        
        keep_pairs.append([new_input, new_output])

    return keep_pairs
