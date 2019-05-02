import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Activation, TimeDistributed, Dense, Embedding, Input,merge,concatenate, GaussianNoise, dot,add
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Layer
from keras.optimizers import Adam
from keras.layers import Dropout, Conv1D, MaxPooling1D, AveragePooling1D
from keras.constraints import maxnorm
from keras.utils import to_categorical
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import deque
from predict_with_features import *

import tensorflow as tf
import os
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Hyper-Parameters
EMBEDDING_DIM = 64
LAYER_NUM = 2
no_filters = 64
HIDDEN_DIM = no_filters * 2
X_max_len = 18
rnn_output_size = 32
Vocabulary_size = 90
NUM_FEATURES = 54
n1, n2, n3, n4, n5, n7, _ = pickle.load(open('./n', 'rb'))

#Loading all encoders-decoders
X_word2idx = pickle.load(open('./X_word2idx', 'rb'))
encoders = pickle.load(open('./phonetic_feature_encoders', 'rb'))
X_idx2word = pickle.load(open('./X_idx2word', 'rb'))
enc = pickle.load(open('./enc', 'rb'))

def encode_words(X):
    X_return = []
    for i, word in enumerate(X):
        temp = []
        for j, char in enumerate(word):
            if char in X_word2idx:
                temp.append(X_word2idx[char])
            else:
                temp.append(X_word2idx['U'])
        X_return.append(temp)
    return X_return

# Converts phonetic_features into encoded form
def encode_features(X_test):
    total_features_to_be_encoded = len(X_test[0][3:])   
    transformed_feature_to_be_returned = []
    for i in range(len(encoders)):
        arr = [w if w in list(encoders[i].classes_) else 'UNK' for w in list(zip(*X_test))[i + 3]]
        transformed_feature_to_be_returned.append(encoders[i].transform(arr))

    X_test = np.asarray(X_test)
    for i in range(total_features_to_be_encoded):
        X_test[:, i + 3] = transformed_feature_to_be_returned[i]
    X_test = X_test.astype(np.float)
    X_test = X_test.tolist()
    return X_test

def getIndexedWords(X_unique):
    X = [list(x) for x in X_unique if len(x) > 0]
    for i, word in enumerate(X):
        for j, char in enumerate(word):
            if char in X_word2idx:
                X[i][j] = X_word2idx[char]
            else:
                X[i][j] = X_word2idx['U']
    return X

# Generate context words
def get_context(X_unique):
    X_left = deque(X_unique)

    X_left.append(' ')  # all elements would be shifted one left
    X_left.popleft()
    X_left1 = list(X_left)
    X_left1 = getIndexedWords(X_left1)
    
    X_left.append(' ')
    X_left.popleft()
    X_left2 = list(X_left)
    X_left2 = getIndexedWords(X_left2)
    
    X_left.append(' ')
    X_left.popleft()
    X_left3 = list(X_left)
    X_left3 = getIndexedWords(X_left3)

    X_left.append(' ')
    X_left.popleft()
    X_left4 = list(X_left)
    X_left4 = getIndexedWords(X_left4)

    X_right = deque(X_unique)

    X_right.appendleft(' ')
    X_right.pop()
    X_right1 = list(X_right)
    X_right1 = getIndexedWords(X_right1)

    X_right.appendleft(' ')
    X_right.pop()
    X_right2 = list(X_right)
    X_right2 = getIndexedWords(X_right2)

    X_right.appendleft(' ')
    X_right.pop()
    X_right3 = list(X_right)
    X_right3 = getIndexedWords(X_right3)

    X_right.appendleft(' ')
    X_right.pop()
    X_right4 = list(X_right)
    X_right4 = getIndexedWords(X_right4)

    return X_left1, X_left2, X_left3, X_left4, X_right1, X_right2, X_right3, X_right4


################################################################################################

def create_model(Vocabulary_size, X_max_len, n_phonetic_features, n1, n2, n3, n4, n5, n6, HIDDEN_DIM, LAYER_NUM):
    def smart_merge(vectors, **kwargs):
        return vectors[0] if len(vectors) == 1 else add(vectors, **kwargs)

    # Declaring the Input Layers
    current_word = Input(shape=(X_max_len,), dtype='float32', name='input1')  # for encoder (shared)
    decoder_input = Input(shape=(X_max_len,), dtype='float32', name='input3')  # for decoder -- attention
    right_word1 = Input(shape=(X_max_len,), dtype='float32', name='input4')
    right_word2 = Input(shape=(X_max_len,), dtype='float32', name='input5')
    right_word3 = Input(shape=(X_max_len,), dtype='float32', name='input6')
    right_word4 = Input(shape=(X_max_len,), dtype='float32', name='input7')
    left_word1 = Input(shape=(X_max_len,), dtype='float32', name='input8')
    left_word2 = Input(shape=(X_max_len,), dtype='float32', name='input9')
    left_word3 = Input(shape=(X_max_len,), dtype='float32', name='input10')
    left_word4 = Input(shape=(X_max_len,), dtype='float32', name='input11')
    phonetic_input = Input(shape=(n_phonetic_features,), dtype='float32', name='input12')

    # Initializing the Character Embeddings
    emb_layer1 = Embedding(Vocabulary_size, EMBEDDING_DIM,
                           input_length=X_max_len,
                           mask_zero=False, name='Embedding')

    list_of_inputs = [current_word, right_word1, right_word2, right_word3, right_word4,
                      left_word1, left_word2, left_word3, left_word4]

    # Calling the embedding Function
    list_of_embeddings = [emb_layer1(i) for i in list_of_inputs]

    #  Dropout Layer
    list_of_embeddings = [Dropout(0.50, name='drop1_' + str(i))(j) for i, j in
                          enumerate(list_of_embeddings)]
    
    # Gaussian Noise Layer
    list_of_embeddings = [GaussianNoise(0.05, name='noise1_' + str(i))(j) for i, j in
                          enumerate(list_of_embeddings)]
                          
    # Applying convolution of filter size 4 for each context word                         
    conv4s  = [Conv1D(filters=no_filters,
                kernel_size=4, padding='valid', activation='relu',
                strides=1, name='conv4_' + str(i))(j) for i, j in enumerate(list_of_embeddings)
            ]

    #Max pooling Operation on each context word
    maxPool4 = [MaxPooling1D(name='max4_' + str(i))(j) for i, j in enumerate(conv4s)]
    
    # Average Pooling Operation on each context word
    avgPool4 = [AveragePooling1D(name='avg4_' + str(i))(j) for i, j in enumerate(conv4s)]

    # Adding the max and average pooling element wise
    pool4s=[add([i, j], name='merge_conv4_' + str(k)) for i, j, k in zip(maxPool4, avgPool4, range(len(maxPool4)))]

    # Repeat Steps for Convolution of filter size 5
    conv5s = [Conv1D(filters=no_filters,
                kernel_size=5,
                padding='valid',
                activation='relu',
                strides=1, name='conv5_' + str(i))(j) for i, j in enumerate(list_of_embeddings)
            ]

    maxPool5 = [MaxPooling1D(name='max5_' + str(i))(j) for i, j in enumerate(conv5s)]
    avgPool5 = [AveragePooling1D(name='avg5_' + str(i))(j) for i, j in enumerate(conv5s)]

    pool5s=[add([i, j], name='merge_conv5_' + str(k)) for i, j, k in zip(maxPool5, avgPool5, range(len(maxPool5)))]
    # End of convolutions for filter size 5

    # Merging the convolutions of both sizes 4 and 5
    mergedPools=pool4s+pool5s

    # Keras Thing for merging
    concat = concatenate(mergedPools, name='main_merge')

    # Dropout Layer
    x = Dropout(0.15, name='drop_single1')(concat)
    
    # Passing the vector of size (None, 7, 1152) generated by concat operation into bidirectional GRU
    x = Bidirectional(GRU(rnn_output_size), name='bidirec1')(x)

    # Adding the phonetic features to RNN's output
    total_features = [x, phonetic_input]
    concat2 = concatenate(total_features, name='phonetic_merging')

    # Two Stacks of fully connected Dense Layers with Dropouts
    x = Dense(HIDDEN_DIM, activation='relu', kernel_initializer='he_normal',
              kernel_constraint=maxnorm(3), bias_constraint=maxnorm(3), name='dense1')(concat2)
    x = Dropout(0.15, name='drop_single2')(x)
    x = Dense(HIDDEN_DIM, kernel_initializer='he_normal', activation='tanh',
              kernel_constraint=maxnorm(3), bias_constraint=maxnorm(3), name='dense2')(x)
    x = Dropout(0.15, name='drop_single3')(x)

    # Prediction Layer with output size same as no. of categories for each tag
    out1 = Dense(n1, kernel_initializer='he_normal', activation='softmax', name='output1')(x)
    out2 = Dense(n2, kernel_initializer='he_normal', activation='softmax', name='output2')(x)
    out3 = Dense(n3, kernel_initializer='he_normal', activation='softmax', name='output3')(x)
    out4 = Dense(n4, kernel_initializer='he_normal', activation='softmax', name='output4')(x)
    out5 = Dense(n5, kernel_initializer='he_normal', activation='softmax', name='output5')(x)
    out6 = Dense(n6, kernel_initializer='he_normal', activation='softmax', name='output6')(x)

    # Luong et al. 2015 attention model
    emb_layer = Embedding(Vocabulary_size, EMBEDDING_DIM,
                          input_length=X_max_len,
                          mask_zero=True, name='Embedding_for_seq2seq')

    current_word_embedding = emb_layer(list_of_inputs[0])
    # current_word_embedding = smart_merge([ current_word_embedding, right_word_embedding1,  left_word_embedding1])

    encoder, state = GRU(rnn_output_size, return_sequences=True, unroll=True, return_state=True, name='encoder')(current_word_embedding)
    encoder_last = encoder[:, -1, :]

    decoder = emb_layer(decoder_input)
    decoder = GRU(rnn_output_size, return_sequences=True, unroll=True, name='decoder')(decoder,initial_state=[encoder_last])

    attention = dot([decoder, encoder], axes=[2, 2], name='dot')
    attention = Activation('softmax', name='attention')(attention)

    context = dot([attention, encoder], axes=[2, 1], name='dot2')
    decoder_combined_context = concatenate([context, decoder], name='concatenate')
    # End of Attention Model 
    
    # Fully Connected Layer before final prediction layer. TimeDistributed is applied because we used sequence to sequence model
    outputs = TimeDistributed(Dense(64, activation='tanh'), name='td1')(decoder_combined_context)

    # Final Prediction Layer. Fully Connected Layer.
    outputs = TimeDistributed(Dense(Vocabulary_size, activation='softmax'), name='td2')(outputs)

    all_inputs = [
                    current_word, decoder_input, right_word1, right_word2,
                    right_word3, right_word4, left_word1,
                    left_word2, left_word3, left_word4, phonetic_input
                ]
    all_outputs = [outputs, out1, out2, out3, out4, out5, out6]

    # Generate the complete keras model.
    model = Model(inputs=all_inputs, outputs=all_outputs)

    return model




def format_output_data(predictions, originals, encoders, pred_features, sentences):

    pred_features[:] = [x.tolist() for x in pred_features]
    for i in range(len(pred_features)):
        pred_features[i] = encoders[i].inverse_transform(pred_features[i])

    f1, f2, f3, f4, f5, f7 = pred_features
    l = []
    for a, b, c, d, e, f, g, h in zip(list(originals), list(predictions), f1, f2, f3, f4, f5, f7):
        l.append([str(a), str(b), str(c), str(d), str(e), str(f), str(g), str(h)])
    return l

# Processes the input to convert it into form feedable to the model. Also parses the output into human readable format
def predict(comment):
    # sentence splitting
    sentences = [line.split() for line in comment.split('\n')]
    global X_max_len, model, n_phonetics, graph
    # List of all words in all sentences
    X_orig = [item for sublist in sentences for item in sublist]

    # Reversing each word
    X_wrds = [item[::-1] for sublist in sentences for item in sublist]
    
    # Converting characters to indices
    X_wrds_inds = encode_words(X_wrds)

    # Generating phonetic features for each word
    X_features = [add_basic_features(sent, word_ind) for sent in sentences for word_ind, _ in enumerate(sent)]
    
    # Convert features to indices
    X_fts = encode_features(X_features)

    # Generate Context Words
    X_left1, X_left2, X_left3, X_left4, X_right1, X_right2, X_right3, X_right4 = get_context(X_wrds)
    
    # Dummy Padding
    X_wrds_inds = pad_sequences(X_wrds_inds, maxlen=X_max_len, dtype='int32', padding='post')
    X_left1 = pad_sequences(X_left1, maxlen=X_max_len, dtype='int32', padding='post')
    X_left2 = pad_sequences(X_left2, maxlen=X_max_len, dtype='int32', padding='post')
    X_left3 = pad_sequences(X_left3, maxlen=X_max_len, dtype='int32', padding='post')
    X_left4 = pad_sequences(X_left4, maxlen=X_max_len, dtype='int32', padding='post')
    X_right1 = pad_sequences(X_right1, maxlen=X_max_len, dtype='int32', padding='post')
    X_right2 = pad_sequences(X_right2, maxlen=X_max_len, dtype='int32', padding='post')
    X_right3 = pad_sequences(X_right3, maxlen=X_max_len, dtype='int32', padding='post')
    X_right4 = pad_sequences(X_right4, maxlen=X_max_len, dtype='int32', padding='post')
    decoder_input = np.zeros_like(X_wrds_inds)
    decoder_input[:, 1:] = X_wrds_inds[:, :-1]
    decoder_input[:, 0] = 1
    scaler = MinMaxScaler()

    # Scaling the phonetic features
    scaler.fit(X_fts)
    X_fts = scaler.transform(X_fts)

    with graph.as_default():
        # Calling the predict function
        words, f1, f2, f3, f4, f5, f7 = model.predict(
            [X_wrds_inds, decoder_input, X_right1, X_right2, X_right3, X_right4, X_left1, X_left2, X_left3,
             X_left4, X_fts])
        
        # Extracting the highest probability prediction for lemma
        predictions = np.argmax(words, axis=2)
        pred_features = [f1, f2, f3, f4, f5, f7]

        # Extracting the highest probability prediction for morphological features
        pred_features = [np.argmax(i, axis=1) for i in pred_features]
        sequences = []

        # Generating the output character sequence from indices
        for i in predictions:
            char_list = []
            for idx in i:
                if idx > 0:
                    char_list.append(X_idx2word[idx])

            sequence = ''.join(char_list)
            sequences.append(sequence)

        # Generating features using the decoders
        data=format_output_data(sequences, X_orig, enc, pred_features, sentences)
    return data


graph = tf.get_default_graph()

if __name__ == "__main__":
    n_phonetics = NUM_FEATURES
    model = create_model(Vocabulary_size, X_max_len, n_phonetics, n1, n2, n3, n4, n5, n7, HIDDEN_DIM, LAYER_NUM)
    model.load_weights('./frozen_training_weights.hdf5')
    f=open('input.txt','r')
    wf=open('output.txt','w')
    sentences=f.readlines()
    for sentence in sentences:
        result=predict(sentence[:-1])  
        for word in result:
            wf.write('\t'.join(word))
            wf.write('\n')
        wf.write('\n')
    f.close()
    wf.close()