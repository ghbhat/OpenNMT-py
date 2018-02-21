#!/bin/sh
#SBATCH --time=1-12
#SBATCH --mem=8g

source activate venv

python translate.py \
-model "/projects/tir1/users/gbhat/work/SMT_en_ru/model_acc_66.98_ppl_4.99_e15.pt" \
-test_path "/projects/tir1/users/gbhat/data/russian/en-ru/" \
-test_src "valid.en" \
-output "/projects/tir1/users/gbhat/work/SMT_en_ru/preds_1" \
-test_tgt "valid.ru" \
-beam_size 5

/usr/bin/perl multi-bleu.perl /projects/tir1/users/gbhat/data/russian/en-ru/valid.ru < \
    /projects/tir1/users/gbhat/work/SMT_en_ru/preds_1