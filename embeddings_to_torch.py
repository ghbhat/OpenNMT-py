#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import six
import sys
import numpy as np
import argparse
import torch
import pdb

parser = argparse.ArgumentParser(description='embeddings_to_torch.py')
parser.add_argument('-emb_file', required=True,
                    help="Embeddings from this file")
parser.add_argument('-output_file', required=True,
                    help="Output file for the prepared data")
parser.add_argument('-dict_file', required=True,
                    help="Dictionary file")
parser.add_argument('-verbose', action="store_true", default=False)
opt = parser.parse_args()


def get_vocab(dict_file):
    vocab = {}
    with open(dict_file, 'r') as f:
        for line in f:
            word, idx = line.split()
            vocab[word.decode('utf8')] = int(idx)

    return vocab

    # vocabs = torch.load(dict_file)
    # enc_vocab, dec_vocab = vocabs[0][1], vocabs[-1][1]

    # print("From: %s" % dict_file)
    # print("\t* source vocab: %d words" % len(enc_vocab))
    # print("\t* target vocab: %d words" % len(dec_vocab))

    # return enc_vocab, dec_vocab


def get_embeddings(file):
    embs = dict()
    for l in open(file, 'rb').readlines():
        try:
            l_split = l.decode('utf8').strip().split()
            if len(l_split) == 2:
                continue
            if len(l_split) == 301:
                embs[l_split[0]] = [float(em) for em in l_split[1:]]
            else:
                embs[l_split[0]] = [float(em) for em in l_split[2:]]
        except:
            print("Error processing %s" % l_split[0])
            continue
    print("Got {} embeddings from {}".format(len(embs), file))

    return embs


def match_embeddings(vocab, emb):
    dim = len(six.next(six.itervalues(emb)))
    filtered_embeddings = np.zeros((len(vocab), dim))
    count = {"match": 0, "miss": 0}
    for w, w_id in vocab.iteritems():
        if w in emb:
            filtered_embeddings[w_id] = emb[w]
            count['match'] += 1
        else:
            if opt.verbose:
                print(u"not found:\t{}".format(w), file=sys.stderr)
            count['miss'] += 1

    return torch.Tensor(filtered_embeddings), count


def main():
    vocab = get_vocab(opt.dict_file)
    print("Read vocab of length %d" % len(vocab))
    embeddings = get_embeddings(opt.emb_file)

    filtered_embeddings, count = match_embeddings(vocab,
                                                    embeddings)

    print("\nMatching: ")
    match_percent = count['match'] / (count['match'] + count['miss']) * 100

    print("\t* embeddings: %d match, %d missing, (%.2f%%)" % (count['match'],
                                                       count['miss'],
                                                       match_percent))

    print("\nFiltered embeddings:")
    print("\t* enc: ", filtered_embeddings.size())

    output_file = opt.output_file + ".pt"
    print("\nSaving embedding as:\n\t* enc: %s"
          % output_file)
    torch.save(filtered_embeddings, output_file)
    print("\nDone.")


if __name__ == "__main__":
    main()
