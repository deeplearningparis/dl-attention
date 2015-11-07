import theano
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
        cs :: word window context size 
        '''
        # parameters of the model
        self.emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
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
        nll = -T.mean(T.log(probas)[y])
        gradients = T.grad(nll, self.params)
        updates = OrderedDict((p, p - lr * g) for p, g in zip(self.params, gradients))
        
        # theano functions
        self.train = theano.function([idxs_enc, idxs_dec, y, lr], nll, updates = updates)


if __name__ == "__main__":

    dim_embedding = 3
    n_classes = 15
    voc_size = 20
    n_hidden = 10

    m = model(nh=n_hidden,
              nc= n_classes + 1, # <eos> token
              ne=voc_size + 1, # <bos> token
              de=dim_embedding)

    # train
    # input :: [1, 2, 3, 4, 5]
    # output :: [-1, 6, 7, 8]
    # target :: [6, 7, 8, -1]
    lr = 0.1

    sentence_enc = np.array(range(6)).astype('int32')
    sentence_dec = np.array([-1] + range(3)).astype('int32')
    target = np.array(range(3) + [-1]).astype('int32')
 
    m.train(sentence_enc, sentence_dec, target, lr)
 
    #TODO add stochasticity
    m.classify(sentence_enc, sentence_dec)
   