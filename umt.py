#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import random

import argparse
import os
import glob
import sys

import torch
import torch.nn as nn
from torch import cuda

import onmt
import onmt.io
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
from onmt.Utils import use_gpu
import opts

from dataproc import *

def parse_args():
    parser = argparse.ArgumentParser(
        description='umt.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.preprocess_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    if opt.word_vec_size != -1:
        opt.src_word_vec_size = opt.word_vec_size
        opt.tgt_word_vec_size = opt.word_vec_size

    if opt.layers != -1:
        opt.enc_layers = opt.layers
        opt.dec_layers = opt.layers

    opt.brnn = (opt.encoder_type == "brnn")

    # if opt.seed > 0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    if torch.cuda.is_available() and not opt.gpuid:
        print("WARNING: You have a CUDA device, should run with -gpuid 0")

    if opt.gpuid:
        cuda.set_device(opt.gpuid[0])
        if opt.seed > 0:
            torch.cuda.manual_seed(opt.seed)

    if len(opt.gpuid) > 1:
        sys.stderr.write("Sorry, multigpu isn't supported yet, coming soon!\n")
        sys.exit(1)

    # Set up the Crayon logging server.
    if opt.exp_host != "":
        from pycrayon import CrayonClient

        cc = CrayonClient(hostname=opt.exp_host)

        experiments = cc.get_experiment_names()
        print(experiments)
        if opt.exp in experiments:
            cc.remove_experiment(opt.exp)


    return opt


def prepare_data(opt):

    print("Initializing languages and vocabularies")
    lang_src = Lang(opt.src_lang)
    lang_tgt = Lang(opt.tgt_lang)
    lang_src.make_vocab(opt.src_vocab, opt.data_path)
    lang_tgt.make_vocab(opt.tgt_vocab, opt.data_path)

    print("Reading parallel training data...")
    train_data = read_parallel_data(opt.data_path, opt.train_src, opt.train_tgt)
    dev_data = read_parallel_data(opt.data_path, opt.valid_src, opt.valid_tgt)
    print("Read %d train and %d dev sequence pairs" % (len(train_data), len(dev_data)))

    print("Filtering sentences by length...")
    train_data = filter_pairs_length(train_data, 0, opt.src_seq_length)
    dev_data = filter_pairs_length(dev_data, 0, opt.src_seq_length)
    print("Retained %d train and %d dev sequence pairs" % (len(train_data), len(dev_data)))

    print("Replacing out-of-vocabulary words...")
    train_data = filter_oov_words(train_data, lang_src, lang_tgt)
    dev_data = filter_oov_words(dev_data, lang_src, lang_tgt)
    print("Done")

    return lang_src, lang_tgt, train_data, dev_data


def build_model(model_opt, opt, lang_src, lang_tgt, checkpoint):
    print('Building model...')
    model = onmt.ModelConstructor.make_base_model(model_opt, lang_src, lang_tgt,
                                                  use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    print(model)

    return model


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def build_optim(opt, model, checkpoint):
    if opt.train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        print('Making optimizer for training.')
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size)

    optim.set_parameters(model.parameters())

    return optim



def make_loss_compute(model, tgt_vocab, opt):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, opt.copy_attn_force)
    else:
        compute = onmt.Loss.NMTLossCompute(
            model.generator, tgt_vocab,
            label_smoothing=opt.label_smoothing)

    if use_gpu(opt):
        compute.cuda()

    return compute


def train_model(model, optim, model_opt, opt, train_data, valid_data, lang_src, lang_tgt):
    train_loss = make_loss_compute(model, lang_tgt.vocab, opt)
    valid_loss = make_loss_compute(model, lang_tgt.vocab, opt)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count

    trainer = onmt.Trainer(model, train_loss, valid_loss, optim,
                           trunc_size, shard_size,
                           norm_method, grad_accum_count)

    print('\nStart training...')
    print(' * number of epochs: %d, starting from Epoch %d' %
          (opt.epochs + 1 - opt.start_epoch, opt.start_epoch))
    print(' * batch size: %d' % opt.batch_size)

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        # 1. Train for one epoch on the training set.
        train_iter = DatasetIterator(train_data, lang_src, lang_tgt, opt.batch_size, opt)
        train_stats = trainer.train(train_iter, epoch, opt, report_func)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        # 2. Validate on the validation set.
        valid_iter = DatasetIterator(valid_data, lang_src, lang_tgt, opt.batch_size, opt)
        valid_stats = trainer.validate(valid_iter)
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # 3. Log to remote server.
        if opt.exp_host:
            train_stats.log("train", experiment, optim.lr)
            valid_stats.log("valid", experiment, optim.lr)

        # 4. Update the learning rate
        trainer.epoch_step(valid_stats.ppl(), epoch)

        # 5. Drop a checkpoint if needed.
        # if epoch >= opt.start_checkpoint_at:
        #     trainer.drop_checkpoint(model_opt, epoch, fields, valid_stats)

class Batch(object):
    def __init__(self, src_var, tgt_var, batch_src_lens, batch_tgt_lens):
        self.src = src_var
        self.tgt = tgt_var
        self.src_lengths = batch_src_lens
        self.tgt_lengths = batch_tgt_lens
        self.batch_size = len(batch_src_lens)

class DatasetIterator(object):
    def __init__(self, dataset, lang_src, lang_tgt, batch_size, opt):
        self.dataset = dataset
        random.shuffle(self.dataset)
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.batch_size = batch_size
        self.num_batches = len(self.dataset)//batch_size + 1
        self.is_cuda = opt.gpuid

    def __iter__(self):
        for i in range(self.num_batches):
            batch_seqs = self.dataset[i*self.batch_size:(i+1)*self.batch_size]
            sorted_batch_seqs = sorted(batch_seqs, key=lambda p: len(p[0]), reverse=True)                             

            batch_src = [pair[0] for pair in sorted_batch_seqs]
            batch_tgt = [pair[1] for pair in sorted_batch_seqs]
            batch_src_lens = [len(seq) for seq in batch_src]
            batch_tgt_lens = [len(seq) for seq in batch_tgt]

            max_src_len = len(batch_src[0])
            max_tgt_len = max(batch_tgt_lens)
            pad_token = self.lang_src.vocab.stoi['<pad>']
            batch_src_padded = [self.pad_seq(seq, max_src_len, pad_token) for seq in batch_src]
            batch_tgt_padded = [self.pad_seq(seq, max_tgt_len, pad_token) for seq in batch_tgt]

            src_var = Variable(torch.LongTensor(batch_src_padded)).transpose(0, 1) 
            tgt_var = Variable(torch.LongTensor(batch_tgt_padded)).transpose(0, 1)
            batch_src_lens = torch.LongTensor(batch_src_lens)
            batch_tgt_lens = torch.LongTensor(batch_tgt_lens)

            if self.is_cuda:
                src_var = src_var.cuda()
                tgt_var = tgt_var.cuda()
                batch_src_lens = batch_src_lens.cuda()
                batch_tgt_lens = batch_tgt_lens.cuda()

            yield Batch(src_var, tgt_var, batch_src_lens, batch_tgt_lens)
        
    def __len__(self):
        return len(self.dataset)

    def pad_seq(self, seq, max_length, PAD_token):                                                                                                   
        seq += [PAD_token for i in range(max_length - len(seq))]                                                            
        return seq  



def report_func(epoch, batch, num_batches,
                start_time, lr, report_stats, opt):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        if opt.exp_host:
            report_stats.log("progress", experiment, lr)
        report_stats = onmt.Statistics()

    return report_stats


def main():

    opt = parse_args()
    
    lang_src, lang_tgt, train_data, valid_data = prepare_data(opt)

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        # I don't like reassigning attributes of opt: it's not clear.
        opt.start_epoch = checkpoint['epoch'] + 1
    else:
        checkpoint = None
        model_opt = opt

    # Build model.
    model = build_model(model_opt, opt, lang_src, lang_tgt, checkpoint)
    tally_parameters(model)
    check_save_model_path(opt)

    # # Build optimizer.
    optim = build_optim(opt, model, checkpoint)

    # # Do training.
    train_model(model, optim, model_opt, opt, train_data, valid_data, lang_src, lang_tgt)




if __name__=="__main__":
    main()
