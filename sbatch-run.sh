#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --time=1-12
#SBATCH --mem=30g

source activate venv

python umt.py -gpuid 0

#/usr/bin/perl multi-bleu.perl /projects/tir1/users/gbhat/data/wmt-splits/test.hi < /projects/tir1/users/gbhat/work/wmt-pred.txt
