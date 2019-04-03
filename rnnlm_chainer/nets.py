#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import division
from __future__ import print_function
import argparse

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import utils


def embed_seq_batch(embed, seq_batch, dropout=0.):
    batchsize = len(seq_batch)
    e_seq_batch = F.split_axis(
        F.dropout(embed(F.concat(seq_batch, axis=0)), ratio=dropout),
        batchsize, axis=0)
    # [(len, ), ] x batchsize
    return e_seq_batch

# Definition of a recurrent net for language modeling


class RNNForLM(chainer.Chain):
    # TODO: nstep LSTM
    def __init__(self, n_vocab, n_units, n_layers=2, dropout=0.5):
        super(RNNForLM, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.rnn = L.NStepLSTM(n_layers, n_units, n_units, dropout)
            # self.output = L.Linear(n_units, n_vocab)
            self.output = L.Linear(n_units, n_vocab)
        del self.output.W
        self.output.W = self.embed.W
        self.dropout = dropout
        self.n_units = n_units
        self.n_layers = n_layers

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.loss = 0.
        self.reset_state()

    def reset_state(self):
        self.h = None
        self.c = None

    def __call__(self, x):
        raise NotImplementedError()

    def call_rnn(self, e_seq_batch):
        batchsize = len(e_seq_batch)
        if self.h is None:
            self.h = self.xp.zeros(
                (self.n_layers, batchsize, self.n_units), 'f')
        if self.c is None:
            self.c = self.xp.zeros(
                (self.n_layers, batchsize, self.n_units), 'f')
        self.h, self.c, y_seq_batch = self.rnn(self.h, self.c, e_seq_batch)
        return y_seq_batch

    def encode_seq(self, x_seq_batch):
        e_seq_batch = embed_seq_batch(
            self.embed, x_seq_batch, dropout=self.dropout)
        y_seq_batch = self.call_rnn(e_seq_batch)
        return y_seq_batch

    def output_from_seq_batch(self, y_seq_batch):
        y = F.concat(y_seq_batch, axis=0)
        y = F.dropout(y, ratio=self.dropout)
        return self.output(y)

    def add_loss(self, y, t, normalize=None):
        loss = F.softmax_cross_entropy(y, t, normalize=False, reduce='mean')
        if normalize is not None:
            loss *= 1. * t.shape[0] / normalize
        else:
            loss *= t.shape[0]
        self.loss += loss

    def add_batch_loss(self, ys, ts, normalize=None):
        if isinstance(ys, (list, tuple)):
            y = F.concat(ys, axis=0)
        else:
            y = ys
        if isinstance(ts, (list, tuple)):
            t = F.concat(ts, axis=0)
        else:
            t = ts
        self.add_loss(y, t, normalize=normalize)

    def pop_loss(self):
        loss = self.loss
        self.loss = 0.
        return loss
