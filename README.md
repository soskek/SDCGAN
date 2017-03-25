# DCGAN-Sentence
This is a sentence generater
using Deep Convolutional Generative Adversarial Network (DCGAN) (http://arxiv.org/abs/1511.06434).
A sentence is represented as sentence-image with shape (`max_sentence_length` x `vector_size`), which is ordered concatenation of word vectors.

This needs dataset of many sentences and pretrained word vectors.

This is derived from [mattya/chainer-DCGAN](https://github.com/mattya/chainer-DCGAN). Thank you!, mattya!


## Generated examples

        0 six woman woman hugging the swimming . <EOS>.7
        1 two men are walking on the at . <EOS>.8
        2 two man and fabrics at the crying near a kitchen . <EOS>.11
        3 decaying young woman is woman are . <EOS>.7
        4 two woman is playing to the pegs . <EOS>.8
        5 the man in just yellow shirt is standing away the water and the sidewalk . <EOS>.15
        6 two men prostitutes are to down a skins them . <EOS>.10
        7 the woman is playing to girl with at her waist-high and mannequins hillock . <EOS>.14
        8 rooting women woman pulling to the crowd with a tugging onlookers . <EOS>.12
        9 two women are playing on a lift . <EOS>.8




