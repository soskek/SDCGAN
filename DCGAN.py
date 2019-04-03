import pickle
import numpy as np
from PIL import Image
import os
import math
import sys
# import codecs
# sys.stdout = codecs.getwriter("utf-8")(sys.stdout)

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer.utils import type_check
from chainer import function
from chainer.initializers import LeCunNormal

import chainer.functions as F
import chainer.links as L

import json
import numpy
import time
from datetime import datetime

image_dir = './images'
out_image_dir = './out_images'
out_model_dir = './out_models'


nz = 512          # # of dim for Z
batchsize = 512
n_epoch = 10000
result_interval = 20
image_save_interval = 100

max_sent = 20
c_units = 32
e_units = c_units // 4
f_units = e_units // 2
g_units = f_units // 2
h_units = g_units // 2

dataset = []
data_file = "coco/dataset.json"
vocab_file_out = "vocab.json"

""" w2v
embed_file = "cbow.512.txt"
embed_vocab = {}
embedW = []
for l in open(embed_file):
    w = l.strip().split()[0]
    v = np.array([float(x) for x in l.strip().split()[1:]], 'f')
    embed_vocab[w] = len(embed_vocab)
    embedW.append(v)

embedW = np.array(embedW)
if '<unk>' not in embed_vocab:
    embed_vocab['<unk>'] = len(embed_vocab)
    embedW = np.concatenate([embedW, embedW.mean(axis=0)[None]], axis=0)
"""

embed_file = "./best.model"
embed_vocab_file = "./best.model.vocab"
# embed_vocab = chainer.datasets.get_ptb_words_vocabulary()
embed_vocab = json.load(open(embed_vocab_file))

UNK_str = "<unk>"
EOS_str = "<eos>"
EOS_id = embed_vocab[EOS_str]
EMPTY_id = len(embed_vocab)

embedW = np.load(embed_file)["embed/W"]
# embedW = np.load(embed_file)["output/W"]
# embedW = (embedW - np.mean(embedW, axis=0)[None])
# embedW = (embedW / np.std(embedW, axis=0)[None])
# embedW = (embedW - np.mean(embedW, axis=1)[:, None])
# embedW = (embedW / np.std(embedW, axis=1)[:, None])
# embedW = F.normalize(embedW).data
# copy EOS vec to EMPTY
embedW = np.concatenate([embedW, embedW[EOS_id:EOS_id + 1, :]], axis=0)
# embedW[EMPTY_id,:] = 0.

embed = L.EmbedID(embedW.shape[0], embedW.shape[1])
embed.W.data[:] = embedW[:]
output = L.Linear(embedW.shape[1], embedW.shape[0], nobias=True)
# output.W.data[:] = embedW[:]
output.W.data[:] = F.normalize(embedW[:]).data
# output.W.data[:] = embedW[:] / np.sqrt(np.sum(embedW[:]**2, axis=1, keepdims=True)+0.0000001)
# output.W.data[:] = embedW[:] / \
#    (np.sum(embedW[:]**2, axis=1, keepdims=True) + 0.0000001)**0.5

# vocab = pickle.load(open(vocab_file))
# vocab = json.load(open(vocab_file))
# rev_vocab = dict((v, k) for k, v in vocab.items())
"""
raw_dataset, raw_dataset2 = pickle.load(open(data_file))
for d in raw_dataset + raw_dataset2:
    if len(d[0]) <= max_sent - 1:
        dataset.append(d[0])
    if len(d[1]) <= max_sent - 1:
        dataset.append(d[1])
"""
vocab = {EOS_str: 0}
dataset = []
for j in json.load(open(data_file))["images"]:
    for s in j["sentences"]:
        for t in s["tokens"]:
            if t not in vocab:
                vocab[t] = len(t)
        if len(s["tokens"]) <= max_sent - 1:
            dataset.append(s["tokens"])
json.dump(vocab, open(vocab_file_out, "w"), indent=2)
new_dataset = []
for d in dataset:
    new_d = [(embed_vocab[d[i]] if d[i] in embed_vocab else embed_vocab[UNK_str])
             if i < len(d) else
             (EOS_id if i == len(d) else EMPTY_id)
             for i in range(max_sent)]
    new_dataset.append(new_d)
vocab = embed_vocab
rev_vocab = dict((v, k) for k, v in vocab.items())

dataset = new_dataset
print(len(dataset))

n_train = len(dataset)


class Refiner(chainer.Chain):
    def __init__(self, u):
        w = chainer.initializers.Normal(0.01)
        super(Refiner, self).__init__(
            c1=L.Convolution2D(
                u, u, (5, 1), stride=(1, 1), pad=(2, 0), initialW=w),
            c2=L.Convolution2D(
                u, u, (5, 1), stride=(1, 1), pad=(2, 0), initialW=w),
            c3=L.Convolution2D(
                u, u, (5, 1), stride=(1, 1), pad=(2, 0), initialW=w),
            bn1=L.BatchNormalization(u),
            bn2=L.BatchNormalization(u),
        )

    def __call__(self, x):
        x += self.c3(F.elu(self.bn2(
            self.c2(F.elu(self.bn1(self.c1(x)))))))
        return x


class Generator(chainer.Chain):

    def __init__(self):
        w = chainer.initializers.Normal(0.01)
        start = max_sent // 2
        super(Generator, self).__init__(
            l0z=L.Linear(nz, start * 128, initialW=w),
            dc1=L.Deconvolution2D(128, 128, (3, 1),
                                  stride=(1, 1), pad=(0, 0), initialW=w),
            dc2=L.Deconvolution2D(128, 256, (3, 1),
                                  stride=(1, 1), pad=(0, 0), initialW=w),
            dc3=L.Deconvolution2D(256, 512, (3, 1),
                                  stride=(1, 1), pad=(0, 0), initialW=w),
            dc4=L.Deconvolution2D(512, 512, (3, 1),
                                  stride=(1, 1), pad=(0, 0), initialW=w),
            dc5=L.Deconvolution2D(512, 512, (3, 1),
                                  stride=(1, 1), pad=(0, 0), initialW=w),
            bn0l=L.BatchNormalization(start * 128),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(256),
            bn3=L.BatchNormalization(512),
            bn4=L.BatchNormalization(512),

            refiner1=Refiner(256),
            refiner2=Refiner(512),
            refiner3=Refiner(512),
        )

    def __call__(self, z, test=False):
        h = F.reshape(F.elu(self.bn0l(self.l0z(z))),
                      (z.data.shape[0], 128, -1, 1))
        h = F.elu(self.bn1(self.dc1(h)))
        h = F.elu(self.bn2(self.dc2(h)))
        h = self.refiner1(h)

        h = F.elu(self.bn3(self.dc3(h)))
        h = self.refiner2(h)

        h = F.elu(self.bn4(self.dc4(h)))
        h = (self.dc5(h))
        h = self.refiner3(h)
        return h


class Discriminator(chainer.Chain):

    def __init__(self):
        w = chainer.initializers.Normal(0.01)
        super(Discriminator, self).__init__(
            c0=L.Convolution2D(512, 512, (3, 1), stride=(
                1, 1), pad=(1, 0), initialW=w),
            c1=L.Convolution2D(512, 512, (3, 1), stride=(
                1, 1), pad=(1, 0), initialW=w),
            c2=L.Convolution2D(512, 512, (3, 1), stride=(
                2, 1), pad=(1, 0), initialW=w),
            c3=L.Convolution2D(512, 512, (3, 1), stride=(
                2, 1), pad=(1, 0), initialW=w),
            # l4l=chainer.Sequential(
            #     L.Linear(None, 512), L.BatchNormalization(512), F.elu, L.Linear(512, 2)),
            l1l=L.Linear(None, 1, initialW=w),
            l2l=L.Linear(None, 1, initialW=w),
            #l3l=L.Linear(None, 1, initialW=w),
            l4l=L.Linear(None, 1, initialW=w),
            bn0=L.BatchNormalization(512),
            bn1=L.BatchNormalization(512),
            bn2=L.BatchNormalization(512),
            bn3=L.BatchNormalization(512),
        )
        self.random_std = 0.2
        self.max_std = 0.5

    def noising(self, x):
        noise = self.xp.random.normal(
            0., self.random_std, x.shape).astype('f')
        scale = self.xp.ones(x.shape).astype('f')
        scale[:, :, :5] = 0.001
        scale[:, :, :7] = 0.01
        scale[:, :, :7:10] = 0.1
        x = x + noise * scale
        return x

    def __call__(self, x):
        x = self.noising(x)
        h0 = F.elu(self.bn0(self.c0(x)))
        with chainer.using_config('train', True):
            h0 = F.dropout(h0, ratio=0.1)
        h1 = F.elu(self.bn1(self.c1(h0)))
        h2 = F.elu(self.bn2(self.c2(h1)))
        h3 = F.elu(self.bn3(self.c3(h2)))
        # h = F.average_pooling_2d((3, 1), stride=(2, 1), pad=(1, 0))
        # h = F.mean(h, axis=2).reshape((x.shape[0], -1))
        # h = F.max(h, axis=2).reshape((x.shape[0], -1))
        l = self.l4l(h3)
        #"""
        with chainer.using_config('train', True):
            # l += self.l3l(F.mean(F.dropout(h2, ratio=0.3), axis=2)
            #              .reshape((h2.shape[0], -1)))
            l += self.l2l(F.dropout(F.mean(h1, axis=2), ratio=0.5)
                          .reshape((h1.shape[0], -1)))
            l += self.l1l(F.dropout(F.mean(h0, axis=2), ratio=0.5)
                          .reshape((h0.shape[0], -1)))
        #"""
        l = F.tanh(l)
        return l


def clip_img(x):
    return np.float32(-1 if x < -1 else (1 if x > 1 else x))


def make_sentences(x):
    x = F.transpose(x, (0, 2, 1, 3))
    h = F.reshape(x, (x.data.shape[0] * max_sent, 512))
    out = output(h)
    arg = xp.argmax(out.data, axis=1)
    seq_L = [[rev_vocab[int(t)] for t in seq]
             for seq in xp.split(arg, x.data.shape[0], axis=0)]
    return seq_L


def train_dcgan_labeled(gen, dis, epoch0=0):
    # o_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
    # o_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
    # o_gen = optimizers.Adam(alpha=0.0001, beta1=0.5)
    # o_dis = optimizers.Adam(alpha=0.0001, beta1=0.5)
    # o_gen = optimizers.Adam(alpha=0.00001, beta1=0.8)
    #o_gen = optimizers.Adam(alpha=0.00003, beta1=0.8)
    #o_dis = optimizers.Adam(alpha=0.00001, beta1=0.8)
    o_gen = optimizers.Adam(alpha=0.0001, beta1=0.8)
    o_dis = optimizers.Adam(alpha=0.00001, beta1=0.8)
    o_gen.setup(gen)
    o_dis.setup(dis)
    # o_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
    # o_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))
    # o_gen.add_hook(chainer.optimizer.WeightDecay(0.000005))
    # o_dis.add_hook(chainer.optimizer.WeightDecay(0.000005))
    o_gen.add_hook(chainer.optimizer.WeightDecay(0.000005))
    o_dis.add_hook(chainer.optimizer.WeightDecay(0.01))
    o_gen.add_hook(chainer.optimizer.GradientClipping(.1))
    o_dis.add_hook(chainer.optimizer.GradientClipping(.1))

    for epoch in range(epoch0, n_epoch):
        perm = np.random.permutation(n_train)
        # sum_l_dis = np.float32(0)
        # sum_l_gen = np.float32(0)
        sum_l_dis = []
        sum_l_gen = []

        accum_dis = 0.
        accum_gen = 0.
        prev_time = time.time()
        dis_result = []  # 1 if dis win gen, 0 otherwise

        for i in range(0, n_train // batchsize):
            update_dis = True  # i % 2
            update_gen = True
            n_ins = len(perm[i * batchsize:(i+1) * batchsize])
            emb_ids = xp.asarray(
                sum([dataset[j] for j in
                     perm[i * batchsize:(i+1) * batchsize]], [])).astype(np.int32)
            x_real = xp.stack(xp.split(embed(emb_ids).data,
                                       batchsize, axis=0), axis=0).transpose((0, 2, 1))[:, :, :, None]
            #x_real = dis.noising(x_real)
            # train generator
            z = Variable(
                xp.random.uniform(-1., 1., (n_ins, nz), dtype=np.float32))
            with chainer.using_config('train', True):
                # x = fill_eos_after_first_eos.
                if update_gen:
                    x_fake = gen(z)
                    #x_fake = dis.noising(x_fake)
                    # with chainer.using_config('train', False):
                    #    y_fake = dis(x_fake)
                    y_fake, y_real = F.split_axis(
                        dis(F.concat([x_fake, x_real], axis=0)), 2, axis=0)
                    #print("f*", y_fake[0:2])
                    L_gen = F.sum(-y_fake) / y_fake.size
                    #print(y_fake[:1], "f*")
                    del y_fake
                    # del x_fake

                # train discriminator
                if update_dis:
                    if update_gen:
                        x_fake = x_fake.data
                    else:
                        x_fake = gen(z).data
                        #x_fake = dis.noising(x_fake)
                    #"""
                    # y_real = dis(x_real)
                    # with chainer.using_config('train', False):
                    # y_fake = dis(x_fake)
                    #"""
                    # y_fake, y_real = F.split_axis(
                    #    dis(F.concat([x_fake, x_real], axis=0)), 2, axis=0)
                    y_fake, y_real = F.split_axis(
                        dis(F.concat([x_fake, x_real], axis=0)), 2, axis=0)
                    #print(y_fake[:1], "f")
                    #print(y_real[:1], "r")
                    L_dis = F.sum(y_fake) / y_fake.size
                    L_dis += F.sum(-y_real) / y_real.size
                    m = (y_fake.data + y_real.data).sum() / \
                        (y_fake.data.size + y_real.data.size)
                    dis_result.extend((y_fake.data < m).tolist())
                    dis_result.extend((m < y_real.data).tolist())
                    # del concat_yl
                    # del concat_x
                    del y_fake
                    del y_real
                    del x_fake
                    del x_real

            if update_gen:
                sum_l_gen.append(L_gen.data)
                accum_gen += float(L_gen.data)
                gen.cleargrads()
                L_gen.backward()
                dis.cleargrads()
                o_gen.update()

            if update_dis:
                sum_l_dis.append(L_dis.data)
                accum_dis += float(L_dis.data)
                dis.cleargrads()
                L_dis.backward()
                gen.cleargrads()
                o_dis.update()

            if epoch >= 1 or i >= 5:
                if WPdis >= 0.8:
                    if dis.random_std * 1.1 < dis.max_std:
                        dis.random_std *= 1.1
                    else:
                        dis.random_std = dis.max_std
                elif WPdis <= 0.2:
                    dis.random_std *= 0.9
                    dis.random_std = np.clip(
                        dis.random_std, 0.01, dis.max_std)
                else:
                    dis.random_std *= 0.995
                    dis.random_std = np.clip(
                        dis.random_std, 0.01, dis.max_std)

            # print "backward done"
            if i % result_interval == 0:
                per = len(dis_result) / batchsize / (time.time() - prev_time)
                prev_time = time.time()
                WPdis = np.mean(dis_result)
                # print(i, "\tWP dis:gen =", WPdis, ":", 1 - WPdis)
                if i > 0:
                    #print(i, "\tWP dis:gen =", WPdis, ":", 1 - WPdis)
                    print('{}\tD:G={:.3f}:{:.3f}\tLD={:.6f},LG={:.6f}\tnoise:{:.6f}\t({:.3f}i/s) {}'.format(
                        i, WPdis, 1-WPdis,
                        accum_dis / (result_interval),
                        accum_gen / (result_interval),
                        dis.random_std, per,
                        datetime.today().strftime("%Y/%m/%d %H:%M:%S")))
                    # print(i, "noise:", dis.random_std,
                    #      "\t(%.3lfi/s)" % per, datetime.today().strftime("%Y/%m/%d %H:%M:%S"))
                    # print(i, "\tLoss dis:", accum_dis / (result_interval / 2),
                    #      "\tgen:", accum_gen / (result_interval / 2))

                accum_gen = 0.
                accum_dis = 0.
                dis_result = []

            if i % image_save_interval == 0:
                z = (xp.random.uniform(-1., 1., (5, nz), dtype=np.float32))
                z = Variable(z)
                with chainer.using_config('train', False):
                    x = gen(z)
                    # x = Variable(x_real[:10])  # debug
                for j, sent_seq in enumerate(make_sentences(x)):
                    sent = []
                    for t in sent_seq:
                        """
                        if t == EOS_str:
                            sent.append("<EOS>."+str(len(sent)))
                            break
                        """
                        if t == EOS_str:
                            t = t.replace(EOS_str, "_")
                        sent.append(t)
                    print("\t", j, " ".join(sent))

        serializers.save_npz("%s/dcgan_model_dis_%d.npz" %
                             (out_model_dir, epoch), dis)
        serializers.save_npz("%s/dcgan_model_gen_%d.npz" %
                             (out_model_dir, epoch), gen)
        serializers.save_npz("%s/dcgan_state_dis_%d.npz" %
                             (out_model_dir, epoch), o_dis)
        serializers.save_npz("%s/dcgan_state_gen_%d.npz" %
                             (out_model_dir, epoch), o_gen)
        # print 'epoch end', epoch, sum_l_gen/n_train, sum_l_dis/n_train
        print('epoch end', epoch, "dis:", sum(sum_l_dis) / len(sum_l_dis) /
              batchsize, "gen:", sum(sum_l_gen) / len(sum_l_gen) / batchsize)
        # dis.max_std *= 0.5


chainer.cuda.get_device_from_id(0).use()
xp = cuda.cupy
# cuda.get_device(0).use()

gen = Generator()
dis = Discriminator()
gen.to_gpu()
dis.to_gpu()
embed.to_gpu()
output.to_gpu()


try:
    os.mkdir(out_image_dir)
    os.mkdir(out_model_dir)
except:
    pass

train_dcgan_labeled(gen, dis)
