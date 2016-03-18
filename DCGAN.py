import pickle
import numpy as np
from PIL import Image
import os
from StringIO import StringIO
import math
#import pylab


import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer.utils import type_check
from chainer import function

import chainer.functions as F
import chainer.links as L


import numpy
import time
from datetime import datetime

image_dir = './images'
out_image_dir = './out_images'
out_model_dir = './out_models'


nz = 200          # # of dim for Z
#batchsize=100
batchsize=64
n_epoch=10000
#n_train=200000
#image_save_interval = 50000
result_interval = batchsize * 20
image_save_interval = batchsize * 100

max_sent = 20
c_units = 32
e_units = c_units / 4
f_units = e_units / 2
g_units = f_units / 2
h_units = g_units / 2
EOS_id = 0
EMPTY_id = 1

# read all images

"""
fs = os.listdir(image_dir)
print len(fs)
dataset = []
for fn in fs:
    f = open('%s/%s'%(image_dir,fn), 'rb')
    img_bin = f.read()
    dataset.append(img_bin)
    f.close()
print len(dataset)
"""
dataset = []
data_file = "/home/sosuke.k/sideline/alignment_entailment/data/dataset.train.pkl"#sys.argv[1]
#data_file = "/home/sosuke.k/sideline/alignment_entailment/data/dataset.test.pkl"#sys.argv[1]
vocab_file = "/home/sosuke.k/sideline/alignment_entailment/data/dataset.vocab.pkl"#sys.argv[2]

embed_file = "/home/sosuke.k/sideline/alignment_entailment/data/model.12.20160318.173857"
embedW = np.load(embed_file)["embed/W"]
embedW[EMPTY_id,:] = 0.
embed = L.EmbedID(embedW.shape[0], embedW.shape[1])
embed.W.data[:] = embedW[:]
output = L.Linear(embedW.shape[1], embedW.shape[0], nobias=True)
output.W.data[:] = embedW[:] / np.sqrt(np.sum(embedW[:]**2, axis=1, keepdims=True)+0.0000001)

vocab = pickle.load(open(vocab_file))
rev_vocab = dict((v,k) for k,v in vocab.items())
raw_dataset, raw_dataset2 = pickle.load(open(data_file))
for d in raw_dataset + raw_dataset2:
    if len(d[0]) <= max_sent-1:
        dataset.append(d[0])
    if len(d[1]) <= max_sent-1:
        dataset.append(d[1])
dataset = [[d[i] if i < len(d) else (EOS_id if i == len(d) else EMPTY_id) for i in range(max_sent)] for d in dataset]
print len(dataset)

n_train=len(dataset)

class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
#            l0z = L.Linear(nz, 6*6*512, wscale=0.02*math.sqrt(nz)),
            l0z = L.Linear(nz, (max_sent-4)*c_units*512, wscale=0.02*math.sqrt(nz)),
            dc1 = L.Deconvolution2D(512, 256, (2,f_units), stride=(1,g_units), pad=(0,h_units), wscale=0.02*math.sqrt(4*4*512)),
            dc2 = L.Deconvolution2D(256, 128, (2,f_units), stride=(1,g_units), pad=(0,h_units), wscale=0.02*math.sqrt(4*4*256)),
            dc3 = L.Deconvolution2D(128, 64, (2,f_units), stride=(1,g_units), pad=(0,h_units), wscale=0.02*math.sqrt(4*4*128)),
            dc4 = L.Deconvolution2D(64, 1, (2,f_units), stride=(1,g_units), pad=(0,h_units), wscale=0.02*math.sqrt(4*4*64)),
            bn0l = L.BatchNormalization((max_sent-4)*c_units*512),
            #bn0 = L.BatchNormalization(512),
            bn1 = L.BatchNormalization(256),
            bn2 = L.BatchNormalization(128),
            bn3 = L.BatchNormalization(64),
        )
        
    def __call__(self, z, test=False):
        h = F.reshape(F.relu(self.bn0l(self.l0z(z), test=test)), (z.data.shape[0], 512, (max_sent-4), c_units))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        x = (self.dc4(h))
        return x



class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            c0 = L.Convolution2D(1, 64, (2,f_units), stride=(1,g_units), pad=(0,h_units), wscale=0.02*math.sqrt(4*4*3)),
            c1 = L.Convolution2D(64, 128, (2,f_units), stride=(1,g_units), pad=(0,h_units), wscale=0.02*math.sqrt(4*4*64)),
            c2 = L.Convolution2D(128, 256, (2,f_units), stride=(1,g_units), pad=(0,h_units), wscale=0.02*math.sqrt(4*4*128)),
            c3 = L.Convolution2D(256, 512, (2,f_units), stride=(1,g_units), pad=(0,h_units), wscale=0.02*math.sqrt(4*4*256)),
            #l4l = L.Linear(max_sent*512, 2, wscale=0.02*math.sqrt(6*6*512)),
            l4l = L.Linear((max_sent-4)*512*c_units, 2, wscale=0.02*math.sqrt(6*6*512)),
            #bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
        )
        
    def __call__(self, x, test=False):
        h = F.elu(self.c0(x))     # no bn because images from generator will katayotteru?
        h = F.dropout(h, ratio=0.05, train=not test)#
        h = F.elu(self.bn1(self.c1(h), test=test))
        h = F.elu(self.bn2(self.c2(h), test=test))
        h = F.elu(self.bn3(self.c3(h), test=test))
        l = self.l4l(h)
        return l




def clip_img(x):
	return np.float32(-1 if x<-1 else (1 if x>1 else x))


def make_sentences(x):
    h = F.reshape(x, (x.data.shape[0]*max_sent, 512))
    out = output(h)
    arg = xp.argmax(out.data, axis=1)
    seq_L = [[rev_vocab[int(t)] for t in seq] for seq in xp.split(arg, x.data.shape[0], axis=0)]
    return seq_L

def train_dcgan_labeled(gen, dis, epoch0=0):
    #o_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
    #o_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_gen = optimizers.Adam(alpha=0.0001, beta1=0.5)
    o_dis = optimizers.Adam(alpha=0.0001, beta1=0.5)
    o_gen.setup(gen)
    o_dis.setup(dis)
    #o_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
    #o_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))
    o_gen.add_hook(chainer.optimizer.WeightDecay(0.000005))
    o_dis.add_hook(chainer.optimizer.WeightDecay(0.000005))

    stop_flag_dis = False
    stop_flag_gen = False
    
    for epoch in xrange(epoch0,n_epoch):
        perm = np.random.permutation(n_train)
        #sum_l_dis = np.float32(0)
        #sum_l_gen = np.float32(0)
        sum_l_dis = []
        sum_l_gen = []

        accum_dis = 0.
        accum_gen = 0.
        prev_time = time.time()
        dis_result = []# 1 if dis win gen, 0 otherwise

        for i in xrange(0, n_train, batchsize):
            # discriminator
            # 0: from dataset
            # 1: from noise
            n_ins = len(perm[i:i+batchsize])

            emb_ids = xp.asarray( sum([dataset[j] for j in perm[i:i+batchsize]], []) ).astype(np.int32)
            x2 = F.reshape( Variable(embed(Variable(emb_ids)).data), (n_ins, 1, max_sent, 512) )
            # reshape de ikeru? soretomo cocat ?
            
            # train generator
            z = Variable(xp.random.uniform(-1, 1, (n_ins, nz), dtype=np.float32))
            x = gen(z)
            yl = dis(x)
            if not stop_flag_gen: L_gen = F.softmax_cross_entropy(yl, Variable(xp.zeros(n_ins, dtype=np.int32)))
            if not stop_flag_dis: L_dis = F.softmax_cross_entropy(yl, Variable(xp.ones(n_ins, dtype=np.int32)))
            
            dis_result.extend([1. if t == 1 else 0. for t in xp.argmax(yl.data, axis=1)])
            
            # train discriminator
            if not stop_flag_dis:
                yl2 = dis(x2)
                L_dis += F.softmax_cross_entropy(yl2, Variable(xp.zeros(n_ins, dtype=np.int32)))
            
            if not stop_flag_gen:
                o_gen.zero_grads()
                L_gen.backward()
                o_gen.update()
                sum_l_gen.append(L_gen.data.get())
                accum_gen += L_gen.data.get()

            if not stop_flag_dis:
                o_dis.zero_grads()
                L_dis.backward()
                o_dis.update()
                sum_l_dis.append(L_dis.data.get())
                accum_dis += L_dis.data.get()

            #print "backward done"
            if i % result_interval == 0:
                per = len(dis_result)*1./(time.time()-prev_time)
                prev_time = time.time()
                print i, "\tdis-train:", not stop_flag_dis, "\tgen-train:", not stop_flag_gen, "\t(%.3lfi/s)" % per, datetime.today().strftime("%Y/%m/%d %H:%M:%S")
                print i, "\tLoss dis:", accum_dis/100/batchsize, "\tgen:", accum_gen/100/batchsize
                WPdis = np.mean(dis_result)
                print i, "\tWP dis:gen =", WPdis, ":", 1-WPdis
                if WPdis >= 0.8 and epoch >= 1:
                    stop_flag_dis = True
                    stop_flag_gen = False
                elif WPdis <= 0.2 and epoch >= 1:
                    stop_flag_dis = False
                    stop_flag_gen = True
                else:
                    stop_flag_dis = False
                    stop_flag_gen = False

                accum_gen = 0.
                accum_dis = 0.
                dis_result = []

            if i % image_save_interval == 0:
                z = (xp.random.uniform(-1, 1, (10, nz), dtype=np.float32))
                z = Variable(z)
                x = gen(z, test=True)
                print "make sentences"
                for j,sent_seq in enumerate(make_sentences(x)):
                    sent = []
                    for t in sent_seq:
                        if t == "EOS":
                            sent.append("<EOS>."+str(len(sent)))
                            break
                        sent.append(t)
                    print "\t",j," ".join(sent)
                
        serializers.save_npz("%s/dcgan_model_dis_%d.npz"%(out_model_dir, epoch),dis)
        serializers.save_npz("%s/dcgan_model_gen_%d.npz"%(out_model_dir, epoch),gen)
        serializers.save_npz("%s/dcgan_state_dis_%d.npz"%(out_model_dir, epoch),o_dis)
        serializers.save_npz("%s/dcgan_state_gen_%d.npz"%(out_model_dir, epoch),o_gen)
        #print 'epoch end', epoch, sum_l_gen/n_train, sum_l_dis/n_train
        print 'epoch end', epoch, "dis:",sum(sum_l_dis)/len(sum_l_dis)/batchsize, "gen:",sum(sum_l_gen)/len(sum_l_gen)/batchsize


xp = cuda.cupy
cuda.get_device(0).use()

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
