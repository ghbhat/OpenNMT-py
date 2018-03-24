#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --time=1-12
#SBATCH --mem=30g

source activate venv

python umt.py -gpuid 0 -data_path "/projects/tir1/users/gbhat/data/russian/en-ru/" \
        -train_tgt "train.en" \
        -train_src "train.ru" \
        -valid_tgt "valid.en" \
        -valid_src "valid.ru" \
        -src_vocab_size 50000 \
        -tgt_vocab_size 50000 \
        -tgt_vocab "/projects/tir1/users/gbhat/work/SMT_en_ru/en_vocab" \
        -src_vocab "/projects/tir1/users/gbhat/work/SMT_en_ru/ru_vocab" \
        -save_model "/projects/tir1/users/gbhat/work/SMT_ru_en/model" \
        -pre_word_vecs_dec "/projects/tir1/users/gbhat/work/SMT_en_ru/wiki.en.pt" \
        -pre_word_vecs_enc "/projects/tir1/users/gbhat/work/SMT_en_ru/wiki.ru.pt" \
        -src_word_vec_size 300 \
        -tgt_word_vec_size 300 \
        -report_every 500 \
        -epochs 20 \
        -batch_size 64 \
        -report_every 3000 \
        -vocab_from_counter
