#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --time=1-12
#SBATCH --mem=30g

source activate venv

# python make_save_vocabs.py -data_path "/projects/tir1/users/gbhat/data/russian/en-ru/" -train_src "train.tokenized.en-ru" -valid_src "dev.en-ru"


# python embeddings_to_torch.py -emb_file /projects/tir1/users/gbhat/data/fasttext/wiki.ru.vec -dict_file /projects/tir1/users/gbhat/work/SMT_en_ru/ru_vocab -output_file /projects/tir1/users/gbhat/work/SMT_en_ru/ru_emb


python umt.py -gpuid 0 -data_path "/projects/tir1/users/gbhat/data/russian/en-ru/" \
        -train_src "train.en" \
        -train_tgt "train.ru" \
        -valid_src "valid.en" \
        -valid_tgt "valid.ru" \
        -src_vocab_size 50000 \
        -tgt_vocab_size 50000 \
        -src_vocab "/projects/tir1/users/gbhat/work/SMT_en_ru/en_vocab" \
        -tgt_vocab "/projects/tir1/users/gbhat/work/SMT_en_ru/ru_vocab" \
        -save_model "/projects/tir1/users/gbhat/work/SMT_en_ru/model" \
        -pre_word_vecs_enc "/projects/tir1/users/gbhat/work/SMT_en_ru/en_emb.pt" \
        -pre_word_vecs_dec "/projects/tir1/users/gbhat/work/SMT_en_ru/ru_emb.pt" \
        -src_word_vec_size 300 \
        -tgt_word_vec_size 300 \
        -report_every 500 \
        -epochs 20 \
        -batch_size 64 \
        -report_every 3000 \
        -vocab_from_counter \
        -layers 2 \
        -learning_rate 0.1 \
        -sgd_momentum 0.95
        

# python translate.py \
# -model "/projects/tir1/users/gbhat/work/SMT_en_ru/model_acc_66.69_ppl_5.12_e10.pt" \
# -test_path "/projects/tir1/users/gbhat/data/russian/en-ru/" \
# -test_src "valid.en" \
# -output "/projects/tir1/users/gbhat/work/SMT_en_ru/preds" \
# -report_bleu \
# -test_tgt "valid.ru" \
# -beam_size 1 \
# -gpu 0 \


# /usr/bin/perl multi-bleu.perl /projects/tir1/users/gbhat/data/russian/en-ru/valid.ru < /projects/tir1/users/gbhat/work/SMT_en_ru/preds