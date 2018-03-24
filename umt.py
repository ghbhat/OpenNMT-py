#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import random, argparse, os, sys

import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable

import onmt
import onmt.io
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
from onmt.Utils import use_gpu
import opts

import gc

from preprocess import *

def parse_args():
    parser = argparse.ArgumentParser(
        description='umt.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.preprocess_opts(parser)
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
            sgd_momentum=opt.sgd_momentum,
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


def train_model(model, optim, model_opt, opt, train_src_seqs, train_tgt_seqs, \
                valid_src_seqs, valid_tgt_seqs, vocab_src, vocab_tgt):
    train_loss = make_loss_compute(model, vocab_tgt, opt)
    valid_loss = make_loss_compute(model, vocab_tgt, opt)

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

    pad_token = vocab_src.stoi['<pad>']

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        
        # 1. Train for one epoch on the training set.
        train_iter = DatasetIterator(train_src_seqs, train_tgt_seqs, pad_token, opt)
        train_stats = trainer.train(train_iter, epoch, opt, report_func)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        # 2. Validate on the validation set.
        valid_iter = DatasetIterator(valid_src_seqs, valid_tgt_seqs, pad_token, opt, \
                        is_inference=True)
        valid_stats = trainer.validate(valid_iter)
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # 3. Update the learning rate
        trainer.epoch_step(valid_stats.ppl(), epoch)

        # 4. Drop a checkpoint if needed.
        if epoch >= opt.start_checkpoint_at:
            trainer.drop_checkpoint(model_opt, vocab_src, vocab_tgt, epoch, valid_stats)

class Batch(object):
    def __init__(self, src, tgt, src_lens, tgt_lens, indices=None, src_maps=None):
        self.src = src
        self.tgt = tgt
        self.src_lengths = src_lens
        self.tgt_lengths = tgt_lens
        self.batch_size = len(src_lens)
        self.indices = indices
        self.src_maps = src_maps


class DatasetIterator(object):
    def __init__(self, src_seqs, tgt_seqs, pad_token, opt, src_maps=None, is_inference=False, is_test=False):
        self.src_seqs = src_seqs
        self.tgt_seqs = tgt_seqs
        self.src_maps = src_maps
        self.pad_token = pad_token
        self.is_cuda = False if is_test else opt.gpuid
        self.is_inference = is_inference
        self.is_test = is_test

        if is_test:
            self.batch_size = opt.batch_size
        else: 
            self.batch_size = opt.valid_batch_size if is_inference else opt.batch_size
        self.num_batches = len(self.src_seqs) // self.batch_size
        if len(self.src_seqs) % self.batch_size != 0:
            self.num_batches += 1
        
    def __iter__(self):
        order = range(self.num_batches)
        if not self.is_test: random.shuffle(order)
        
        for i in order:
            start = i*self.batch_size
            ln = self.batch_size
            if start+ln>len(self.src_seqs):
                ln = len(self.src_seqs)-start

            src = self.src_seqs[start:start+ln]
            tgt = self.tgt_seqs[start:start+ln]
            src_lens = [len(seq) for seq in src]
            tgt_lens = [len(seq) for seq in tgt]
    
            if self.is_test:
                indices = range(start, start+ln)
                src_maps = self.src_maps[start:start+ln]
                sorted_materials = sorted(zip(src, tgt, src_lens, tgt_lens, indices, src_maps), 
                                    key=lambda p: p[2], reverse=True)
                src, tgt, src_lens, tgt_lens, indices, src_maps = zip(*sorted_materials)
            else:
                sorted_materials = sorted(zip(src, tgt, src_lens, tgt_lens), 
                                    key=lambda p: p[2], reverse=True)
                src, tgt, src_lens, tgt_lens = zip(*sorted_materials)
            
            max_src_len = src_lens[0]
            max_tgt_len = max(tgt_lens)
            src_padded = [pad_seq(seq, max_src_len, self.pad_token) for seq in src]
            tgt_padded = [pad_seq(seq, max_tgt_len, self.pad_token) for seq in tgt]

            src_var = Variable(torch.LongTensor(src_padded).transpose(0,1), \
                volatile=self.is_inference)
            tgt_var = Variable(torch.LongTensor(tgt_padded).transpose(0,1), \
                volatile=self.is_inference)
            # if not self.is_test:
            #     tgt_var = Variable(tgt_var, volatile=self.is_inference)
            # print(src_var.size())
            src_lens = torch.LongTensor(src_lens)
            tgt_lens = torch.LongTensor(tgt_lens)

            if self.is_cuda:
                src_var = src_var.cuda()
                tgt_var = tgt_var.cuda()
                src_lens = src_lens.cuda()
                tgt_lens = tgt_lens.cuda()

            if self.is_test:
                indices = Variable(torch.LongTensor(indices), volatile=self.is_inference)
                src_maps = self.make_src(src_maps)
                if self.is_cuda:
                    indices = indices.cuda()
                    src_maps = src_maps.cuda()
                yield Batch(src_var, tgt_var, src_lens, tgt_lens, indices, src_maps)
            else:
                yield Batch(src_var, tgt_var, src_lens, tgt_lens)
        
    def __len__(self):
        return self.num_batches

    def pad_seq(self, seq, max_length, PAD_token):                                                                                                   
        seq += [PAD_token for i in range(max_length - len(seq))]
        return seq

    #  From TextDataset.py
    def make_src(self, data):
        src_size = max([t.size(0) for t in data])
        src_vocab_size = max([t.max() for t in data]) + 1
        alignment = torch.zeros(src_size, len(data), src_vocab_size)
        for i, sent in enumerate(data):
            for j, t in enumerate(sent):
                alignment[j, i, t] = 1
        return alignment
  

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
    print(opt)
    
    train_src_seqs, train_tgt_seqs, valid_src_seqs, valid_tgt_seqs, \
    src_vocab, tgt_vocab = prepare_train_data(opt)

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
    model = build_model(model_opt, opt, src_vocab, tgt_vocab, checkpoint)
    tally_parameters(model)
    check_save_model_path(opt)

    # # Build optimizer.
    optim = build_optim(opt, model, checkpoint)

    # # Do training.
    train_model(model, optim, model_opt, opt, \
                train_src_seqs, train_tgt_seqs, \
                valid_src_seqs, valid_tgt_seqs, \
                src_vocab, tgt_vocab)




if __name__=="__main__":
    main()
