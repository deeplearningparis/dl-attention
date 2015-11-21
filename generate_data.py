# -*- coding: utf-8 -*-

# generates addition dataset
# taken from Keras (https://github.com/fchollet/keras/blob/master/examples/addition_rnn.py)

import numpy as np
from six.moves import range

INVERT = False
DIGITS = 3
MAXLEN = DIGITS + 1 + DIGITS

# Not needed (we will use indexes)


class CharacterTable(object):
    """
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    """
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode_index(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen), dtype='int32')
        for i, c in enumerate(C):
            X[i] = self.char_indices[c]
        return X

    def decode_index(self, X):
        return ''.join(self.indices_char[x] for x in X)

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)

chars = '0123456789+ '
ctable = CharacterTable(chars, MAXLEN)


def generate_train_data(training_size=5000):
    questions = []
    expected = []
    seen = set()
    print('Generating data...')
    while len(questions) < training_size:
        f = lambda: int(''.join(np.random.choice(list('0123456789'))
                        for i in range(np.random.randint(1, DIGITS + 1))))
        a, b = f(), f()
        # Skip any addition questions we've already seen
        # Also skip any such that X+Y == Y+X (hence the sorting)
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        # Pad the data with spaces such that it is always MAXLEN
        q = '{}+{}'.format(a, b)
        query = q + ' ' * (MAXLEN - len(q))
        ans = str(a + b)
        # Answers can be of maximum size DIGITS + 1
        ans += ' ' * (DIGITS + 1 - len(ans))
        if INVERT:
            query = query[::-1]
        questions.append(query)
        expected.append(ans)
        #  print('Total addition questions:', len(questions))

    print('Vectorization...')
    X = np.zeros((len(questions), MAXLEN), dtype='int32')
    y = np.zeros((len(questions), DIGITS + 1), dtype='int32')
    for i, sentence in enumerate(questions):
        X[i] = ctable.encode_index(sentence, maxlen=MAXLEN)
        #  X[i] = ctable.encode(sentence, maxlen=MAXLEN)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode_index(sentence, maxlen=DIGITS + 1)
        #  y[i] = ctable.encode(sentence, maxlen=DIGITS + 1)

    # Shuffle (X, y) in unison as the later parts of X
    # will almost all be larger digits
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    # Explicitly set apart 10% for validation data that we never train over
    split_at = len(X) - len(X) / 10
    (X_train, X_val) = (X[0:split_at, ], X[split_at:, ])
    (y_train, y_val) = (y[:split_at], y[split_at:])

    print("X_train shape:" + str(X_train.shape))
    print("y_train shape:" + str(y_train.shape))
    return X_train, X_val, y_train, y_val
