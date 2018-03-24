from __future__ import unicode_literals, print_function, division
import os, codecs, torchtext, torch, argparse
from collections import Counter
import opts
import codecs

def parse_args():
    parser = argparse.ArgumentParser(
        description='umt.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    opts.add_md_help_argument(parser)
    opts.preprocess_opts(parser)
    opt = parser.parse_args()
    return opt


def read_vocab(filename, data_path):
    vocab_file = os.path.join(data_path, filename)
    vocab = codecs.open(vocab_file, encoding='utf-8').read().strip().split('\n')
    vocab_counter = Counter()
    for word in vocab:
      vocab_counter[word] += 1
    return vocab_counter


def load_vocab_counter(dict_file):
    vocab_counter = Counter()                                                                                                          
    with open(dict_file, 'r') as f:                                                                                     
        for line in f:                                                                                                  
            word, idx = line.split()                                                                                    
            vocab_counter[word.decode('utf8')] = int(idx)                                                                       
                                                                                              
    return vocab_counter


def make_vocab(filename, data_path, from_counter=False):
    if from_counter:
        vocab_counter = load_vocab_counter(filename)
    else:
        vocab_counter = read_vocab(filename, data_path)
    specials = ['<unk>', '<pad>', '<sos>', '<eos>']
    return torchtext.vocab.Vocab(vocab_counter, specials=specials)


def read_parallel_data(data_path, source_filename, target_filename):
    # Read the file and split into lines
    source_path = os.path.join(data_path, source_filename)
    target_path = os.path.join(data_path, target_filename)
    source_seqs = codecs.open(source_path, encoding='utf-8').read().strip().split('\n')
    target_seqs = codecs.open(target_path, encoding='utf-8').read().strip().split('\n')
    
    # Normalize sentences, make into pairs and sort in descending order
    source_seqs = [line.lower().strip().split() for line in source_seqs]
    target_seqs = [line.lower().strip().split() for line in target_seqs]

    return source_seqs, target_seqs


def filter_pairs_length(pairs, min_length, max_length):
    filtered_pairs = []
    for pair in pairs:
        if len(pair[0]) >= min_length and len(pair[0]) <= max_length \
            and len(pair[1]) >= min_length and len(pair[1]) <= max_length:
                filtered_pairs.append(pair)
    return filtered_pairs


def filter_oov_and_bound(sentence, vocab):
    filtered_sentence = []
    # filtered_sentence.append(vocab.stoi['<sos>'])
    for word in sentence:
        if word not in vocab.stoi:
            filtered_sentence.append(vocab.stoi['<unk>'])
        else:
            filtered_sentence.append(vocab.stoi[word])
    filtered_sentence.append(vocab.stoi['<eos>'])
    return filtered_sentence


def pad_seq(seq, max_length, PAD_token):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def make_seq_map(seqs):
    seq_vocabs = []
    seq_maps = []
    
    for seq in seqs:
        seq_vocab = torchtext.vocab.Vocab(Counter(seq))
        seq_vocabs.append(seq_vocab)
        # Mapping source tokens to indices in the dynamic dict.
        seq_map = torch.LongTensor([seq_vocab.stoi[w] for w in seq])
        seq_maps.append(seq_map)

    return seq_vocabs, seq_maps
    

def prepare_train_data(opt):

    print("Initializing languages and vocabularies")
    src_vocab = make_vocab(opt.src_vocab, opt.data_path, opt.vocab_from_counter)
    tgt_vocab = make_vocab(opt.tgt_vocab, opt.data_path, opt.vocab_from_counter)

    print("Reading parallel training data...")
    train_src, train_tgt = read_parallel_data(opt.data_path, opt.train_src, opt.train_tgt)
    dev_src, dev_tgt = read_parallel_data(opt.data_path, opt.valid_src, opt.valid_tgt)
    print("Done")
    print("Read %d train and %d dev sequence pairs" % (len(train_src), len(dev_src)))
    print()

    print("Filtering sentences by length...")
    train_data = filter_pairs_length(zip(train_src, train_tgt), 0, opt.src_seq_length)
    dev_data = filter_pairs_length(zip(dev_src, dev_tgt), 0, opt.src_seq_length)
    train_src, train_tgt = zip(*train_data)
    dev_src, dev_tgt = zip(*dev_data)
    print("Done")
    print("Retained %d train and %d dev sequence pairs" % (len(train_src), len(dev_src)))
    print()

    print("Replacing out-of-vocabulary words and bounding...")
    train_src = [filter_oov_and_bound(seq, src_vocab) for seq in train_src]
    train_tgt = [filter_oov_and_bound(seq, tgt_vocab) for seq in train_tgt]
    dev_src = [filter_oov_and_bound(seq, src_vocab) for seq in dev_src]
    dev_tgt = [filter_oov_and_bound(seq, tgt_vocab) for seq in dev_tgt]
    print("Done")
    print()

    return train_src, train_tgt, dev_src, dev_tgt, src_vocab, tgt_vocab

    
def prepare_test_data(opt, vocab_src, vocab_tgt):
    
    print("Reading test data...")
    raw_src, raw_tgt = read_parallel_data(opt.test_path, opt.test_src, opt.test_tgt)
    print("Read %d test sequence pairs" % len(raw_src))
    print()

    print("Replacing out-of-vocabulary words and bounding...")
    src = [filter_oov_and_bound(seq, vocab_src) for seq in raw_src]
    tgt = [filter_oov_and_bound(seq, vocab_tgt) for seq in raw_tgt]
    print("Done")
    print()

    seq_vocabs, seq_maps = make_seq_map(src)

    return raw_src, src, tgt, seq_vocabs, seq_maps
    

if __name__ == "__main__":
    main()