import theano
import sys
from generate_data import generate_train_data, CharacterTable
import pdb
import numpy
import numpy as np
import os

from theano import tensor as T
from collections import OrderedDict

class model(object):
    
    def __init__(self, nh, nc, ne, de):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        '''
        # parameters of the model
        self.emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (ne, de)).astype(theano.config.floatX))
        self.Wx_enc  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de, nh)).astype(theano.config.floatX))
        self.Wx_dec  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de, nh)).astype(theano.config.floatX))
        self.Wh_enc  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.Wh_dec  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        
        self.h0_enc  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.h0_dec  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))

        self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nc)).astype(theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))

        # bundle
        self.params = [self.emb, self.Wx_enc, self.Wx_dec, self.Wh_enc, self.Wh_dec, self.W, self.b, self.h0_enc, self.h0_dec]

        idxs_enc, idxs_dec = T.ivector(), T.ivector() 
        y = T.ivector()
        x_enc, x_dec = self.emb[idxs_enc], self.emb[idxs_dec]
        
        # compute the encoder representation
        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx_enc) + T.dot(h_tm1, self.Wh_enc))
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence, \
            sequences=x_enc, outputs_info=[self.h0_enc, None])
        h_enc = h[-1, :]
        
        # from the encoder representation, generate the sequence 
        
        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx_dec) + T.dot(h_tm1, self.Wh_dec) + h_enc)
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            return [h_t, s_t]

        [h_dec, s_dec], _ = theano.scan(fn=recurrence, \
            sequences=x_dec, outputs_info=[self.h0_dec, None])
     
        probas = s_dec[:, 0, :]
        y_pred = T.argmax(probas, axis=1)

        self.classify = theano.function(inputs=[idxs_enc, idxs_dec], outputs=y_pred)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -T.mean(T.log(probas)[T.arange(y.shape[0]), y])
        gradients = T.grad(nll, self.params)
        updates = OrderedDict((p, p - lr * g) for p, g in zip(self.params, gradients))
        
        # theano functions
        self.train = theano.function([idxs_enc, idxs_dec, y, lr], nll, updates=updates)

def preprocess(x, y):
    x, y = filter(lambda z: z != 0, x), filter(lambda z: z != 0, y)
    sentence_enc = np.array(x).astype('int32')
    sentence_dec = np.array([0] + y[:-1]).astype('int32') - 1 # trick with 1-based indexing
    target = np.array(y[0] + [-1]).astype('int32') - 1 # same
    return sentence_enc, sentence_dec, target

def main(nsamples=100,
         dim_embedding=15,
         n_hidden=128,
         lr=0.1,
         nepochs=10):

    INVERT = False
    DIGITS = 3
    MAXLEN = DIGITS + 1 + DIGITS
    chars = '0123456789+ '
    n_classes = len('0123456789') + 1 # add <eos>
    voc_size = len('0123456789+') + 1 # add <bos> for the decoder 

    # generate the dataset
    ctable = CharacterTable(chars, MAXLEN)
    X_train, X_val, y_train, y_val = generate_train_data(nsamples) 

    # build the model
    m = model(nh=n_hidden,
              nc=n_classes, 
              ne=voc_size, 
              de=dim_embedding)

    # training
    for epoch in range(nepochs):
        for i, (x, y) in enumerate(zip(X_train, y_train)):
            sentence_enc, sentence_dec, target = preprocess(x, y)
            m.train(sentence_enc, sentence_dec, target, lr)
            print "%.2f %% completed\r" % ((i + 1) * 100. / nsamples), 
            sys.stdout.flush()
        print

if __name__ == "__main__":
    main()
