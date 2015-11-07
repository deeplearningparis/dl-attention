import numpy as np


# Return the generated text produced by the decoder
def generate(pre_softmax, inputs, targets, data, max_generation_length, batch_size, vocab_size):
    # pre_softmax   : Theano variable : the pre_softmax value
    # targets   : Theano variable : the target sequence (right side of the equation)
    # input    : Theano variable : the input sequence (left side of the equation)
    # data          : Numpy array :  the values of the data (left part) with shape (Time X Batch X Vocab_size)
    # max_generation_lenght : int : the max number of the output
    # batch_size : int
    # vocab_size : int

    generate_step = theano.function(
        inputs=[inputs, targets], outputs=pre_softmax)

    # The text that has been generated (Time X Batch)
    generated_text = np.zeros((0, batch_size))

    # The probability array (Time X Batch X vocab_size)
    probability_array = np.zeros((0, batch_size, vocab_size))

    for i in range(max_generation_length):
        presoft = generate_step([data, generated_text])

        # Get the value of the last pre_softmax
        last_pre_softmax = presoft[-1:, :, :]

        # Compute the probability distribution
        probabilities = softmax(last_pre_softmax)

        # Add it in the probabily array
        probability_array = np.vstack([probability_array, probabilities])

        # Sample a character out of the probability distribution
        # TODO change argmax to a sample
        sampled_character = sample(probabilities, axis=2)

        # Concatenate the new value to the text
        generated_text = np.vstack([generated_text, sampled_character])

    return generated_text


# python softmax
def softmax(w):
    e = np.exp(w)
    dist = e / np.sum(e, axis=1)
    return dist


# sample
def sample(probabilities):
    bins = np.add.accumulate(probabilities[0])
    return np.digitize(np.random.random_sample(1), bins)
