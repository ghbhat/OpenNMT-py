import torch, codecs
from umt_en_ru import *

opt = parse_args()

train_src, train_tgt, dev_src, dev_tgt, src_vocab, tgt_vocab = prepare_train_data(opt)

with codecs.open("/projects/tir1/users/gbhat/work/SMT_en_ru/en_vocab", 'w', encoding='utf-8') as file:
    count = 0
    for key, val in src_vocab.stoi.iteritems():
        file.write('%s %d\n' % (key, val))
        count += 1
    print(count)
file.close()

with codecs.open("/projects/tir1/users/gbhat/work/SMT_en_ru/ru_vocab", 'w', encoding='utf-8') as file:
    count = 0
    for key, val in tgt_vocab.stoi.iteritems():
        file.write('%s %d\n' % (key, val))
        count += 1
    print(count)
file.close()
