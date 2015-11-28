import theano
import sys
from generate_data import generate_train_data, CharacterTable
import pdb
import numpy
import numpy as np
import os

from theano import tensor as T
from collections import OrderedDict

theano.config.mode = 'FAST_COMPILE'


class model(object):
    def __init__(self, nh, nc, ne, natt, batch_size=64, attention_type='no_attention'):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        natt :: dimension of hidden attention layer
        '''
        self.nh = nh
        self.ne = ne
        # parameters of the model
        self.Wx_enc  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (ne, nh)).astype(theano.config.floatX))
        self.Wx_dec  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (ne, nh)).astype(theano.config.floatX))

        self.h0_enc  = theano.shared(numpy.zeros((batch_size,nh), dtype=theano.config.floatX))
        self.h0_dec  = theano.shared(numpy.zeros((batch_size,nh), dtype=theano.config.floatX))

        self.Wh_enc   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                                                                 (nh, nh)).astype(theano.config.floatX))
        self.Wh_dec   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                                                                 (nh, nh)).astype(theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
        self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                                                                 (nh, nc)).astype(theano.config.floatX))
        # bundle
        self.params = [self.Wx_enc, self.Wx_dec, self.Wh_enc, self.Wh_dec, self.W, self.b, self.h0_enc, self.h0_dec]

        if attention_type == 'dnn':
            self.natt = natt
            self.W_att_enc  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                    (nh, natt)).astype(theano.config.floatX))
            self.W_att_dec  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                    (nh, natt)).astype(theano.config.floatX))
            self.W_att_out  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (natth)).astype(theano.config.floatX))
            self.params += [self.W_att_enc, self.W_att_dec, self.W_att_out]

        idxs_enc, idxs_dec = T.imatrix(), T.imatrix() 
        y = T.imatrix()
        # shape (batch, seq, dim)
        x_enc, x_dec = self.Wx_enc[idxs_enc], self.Wx_dec[idxs_dec]
        
        # compute the encoder representation
        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(x_t + T.dot(h_tm1, self.Wh_enc))
            return [h_t, h_t]
        
        [h, s], _ = theano.scan(fn=recurrence, \
                                sequences=x_enc.dimshuffle(1,0,2), \
                                outputs_info=[self.h0_enc, None])
        h_enc_last = h[-1, :]
        # shape of h: (seq, batch, dim)

        # No attention : return the last element of h_enc
        def no_attention(h_enc, h_tm1):
            return h_enc[-1, :]

        # Simple MemNN style attention = similarity between h_enc and h_tm1
        def attention_function_dot(h_enc, h_tm1):
            attention_vector = T.nnet.softmax(T.dot(h_enc, h_tm1))
            return (attention_vector.T * h_enc).sum(axis=0) 

        # TODO Attention computed with an NN (1 hidden layer for states mixing)
        def attention_function_dnn(h_enc, h_tm1):
            attn_hid = T.tanh(T.dot(h_enc, self.W_att_enc) + T.dot(h_tm1, self.W_att_dec))            
            attention_vector = T.nnet.softmax(T.dot(attn_hid, self.W_att_out.T))
            return (attention_vector.T * h_enc).sum(axis=0) 

        if attention_type == 'dnn':
            attention = attention_function_dnn
        elif attention_type == 'dot':
            attention = attention_function_dot
        else:
            attention = no_attention

        # from the encoder representation, generate the sequence 
        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(x_t + T.dot(h_tm1, self.Wh_dec) + attention(h, h_tm1))
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            return [h_t, s_t]

        [h_dec, s_dec], _ = theano.scan(fn=recurrence, \
                                        sequences=x_dec.dimshuffle(1,0,2), 
                                        outputs_info=[self.h0_dec, None])
     
        probas = s_dec.dimshuffle(1,0,2)
        y_pred = T.argmax(probas, axis=2)

        self.classify = theano.function(inputs=[idxs_enc, idxs_dec], outputs=y_pred)
        self.debug = theano.function(inputs=[idxs_enc, idxs_dec, y], outputs=[idxs_enc.shape, y.shape, x_enc.shape, h.shape, h_enc_last.shape, h_dec.shape, s_dec.shape, probas.shape, y_pred.shape])

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        #nll = -T.mean(T.log(probas)[T.arange(y.shape[0]), y])
        #nll = -T.mean(y * T.log(probas)+ (1.- y) * T.log(1. - probas))
        gradients = T.grad(nll, self.params)
        updates = OrderedDict((p, p - lr * g) for p, g in zip(self.params, gradients))
        
        # theano functions
        self.train = theano.function([idxs_enc, idxs_dec, y, lr], nll, updates=updates)
        
        # generation part 
        h_tm1 = T.vector()
        idxs_dec = T.iscalar() 

        h_t = T.nnet.sigmoid(self.Wx_dec[idxs_dec] + T.dot(h_tm1, self.Wh_dec) + attention(h, h_tm1))
        s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)

        self.compute_h_enc = theano.function([idxs_enc], h)
        self.generate_step = theano.function(inputs=[h_tm1, h, idxs_dec], outputs=[h_t, s_t])



    def generate_text(self, idxs_enc, max_len = 10):
        h_T = self.compute_h_enc(idxs_enc) 
        cur_dec_idx = -1
        y_pred = []
        for i in range(max_len):
            if i == 0:
                h_tm1 = self.h0_dec.get_value()
            h_tm1, probas = self.generate_step(h_tm1, h_T, cur_dec_idx)
            # sample given the multinomial
            cur_dec_idx = np.argwhere(numpy.random.multinomial(1, probas[0]) == 1)[0][0]
            y_pred += [cur_dec_idx]
            if cur_dec_idx == len(probas[0]) - 1:
                # we sampled <EOS>
                break
        return y_pred

def preprocess(x, y):
    #x, y = filter(lambda z: z != 0, x), filter(lambda z: z != 0, y)
    sentence_enc = np.array(x).astype('int32')
    sentence_dec = np.array([0] + y).astype('int32') - 1 # trick with 1-based indexing
    target = np.array(y + [0]).astype('int32') - 1 # same
    return sentence_enc, sentence_dec, target

def main(nsamples=10000,
         n_hidden=128,
         lr=0.01,
         nepochs=100,
         batch_size=64,
         val_freq=1):

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
              batch_size=batch_size,
              natt=20)

    b_sentence_enc = np.zeros((batch_size,MAXLEN)).astype('int32')
    b_sentence_dec = np.zeros((batch_size,DIGITS + 2)).astype('int32')
    b_target = np.zeros((batch_size,DIGITS + 2)).astype('int32')
    print(m.debug(b_sentence_enc,b_sentence_dec,b_target))
    # training
    for epoch in range(nepochs):
        nlls = []
        for batch_num in range(len(X_train) / batch_size):
            b_sentence_enc = np.zeros((batch_size,MAXLEN)).astype('int32')
            b_sentence_dec = np.zeros((batch_size,DIGITS + 1)).astype('int32')
            b_target = np.zeros((batch_size,DIGITS + 1)).astype('int32')
            for i in range(batch_size):
                x, y = X_train[batch_num*batch_size + i], y_train[batch_num*batch_size + i]
                sentence_enc, sentence_dec, target = preprocess(x, y)
                b_sentence_enc[i,] = sentence_enc
                b_sentence_dec[i,] = sentence_dec
                b_target[i,] = target                        
                nlls += [m.train(b_sentence_enc, b_sentence_dec, b_target, lr)]

            print "%.2f %% completedi - nll = %.2f\r" % ((i + 1) * 100. / len(X_train), np.mean(nlls)), 
            sys.stdout.flush()
        print

        # evaluation
        if (epoch + 1) % val_freq == 0: 
            for i, (x, y) in enumerate(zip(X_val, y_val)):
                sentence_enc, sentence_dec, target = preprocess(x, y)
                y_pred = m.generate_text(sentence_enc)
                try:
                    print "ground-truth\t", np.concatenate([[sentence_dec[1]], target[:-1]])
                    print "predicted   \t", y_pred
                except IndexError:
                    pass
                if i > 5:
                    break

if __name__ == "__main__":
    main()
