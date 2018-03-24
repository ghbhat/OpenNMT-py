#!/bin/sh
#SBATCH --time=1-12
#SBATCH --mem=8g

source activate venv

python translate.py \
-model "/projects/tir1/users/gbhat/work/SMT_ru_en/model_acc_62.52_ppl_7.45_e20.pt" \
-test_path "/projects/tir1/users/gbhat/data/russian/en-ru/" \
-test_src "valid.ru" \
-output "/projects/tir1/users/gbhat/work/SMT_ru_en/preds" \
-test_tgt "valid.en" \
-beam_size 5

/usr/bin/perl multi-bleu.perl /projects/tir1/users/gbhat/data/russian/en-ru/valid.en < \
    /projects/tir1/users/gbhat/work/SMT_ru_en/preds