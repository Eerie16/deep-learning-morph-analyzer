from make_prediction import *
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NUMBER_OF_INPUTS = 11
TRAINING_SIZE = 100
counter =0
feature_encoders = pickle.load(open('feature-encoders.pickle','rb'))
input_dict={
    'input1':np.zeros(shape=(TRAINING_SIZE, X_max_len)),
    'input2':np.zeros(shape=(TRAINING_SIZE, X_max_len)),
    'input3':np.zeros(shape=(TRAINING_SIZE, X_max_len)),
    'input4':np.zeros(shape=(TRAINING_SIZE, X_max_len)),
    'input5':np.zeros(shape=(TRAINING_SIZE, X_max_len)),
    'input6':np.zeros(shape=(TRAINING_SIZE, X_max_len)),
    'input7':np.zeros(shape=(TRAINING_SIZE, X_max_len)),
    'input8':np.zeros(shape=(TRAINING_SIZE, X_max_len)),
    'input9':np.zeros(shape=(TRAINING_SIZE, X_max_len)),
    'input10':np.zeros(shape=(TRAINING_SIZE, X_max_len)),
    'input11':np.zeros(shape=(TRAINING_SIZE, NUM_FEATURES)),
}
output_dict={
    'td2':np.zeros(shape=(TRAINING_SIZE, X_max_len, Vocabulary_size)),
    'output1':np.zeros(shape=(TRAINING_SIZE, X_max_len, n1)),
    'output2':np.zeros(shape=(TRAINING_SIZE, X_max_len, n2)),
    'output3':np.zeros(shape=(TRAINING_SIZE, X_max_len, n3)),
    'output4':np.zeros(shape=(TRAINING_SIZE, X_max_len, n4)),
    'output5':np.zeros(shape=(TRAINING_SIZE, X_max_len, n5)),
    'output6':np.zeros(shape=(TRAINING_SIZE, X_max_len, n7)),
}
output_arr_words=[]
X_word_to_idx = pickle.load(open('./X_word2idx',    'rb'))

def generate_input(sentence):
    X_wrds = [item[::-1] for item in sentence]
    X_wrds_inds = encode_words(X_wrds)
    X_features = [add_basic_features(sentence, word_ind) for word_ind, _ in enumerate(sentence)]
    X_fts = encode_features(X_features)
    X_left1, X_left2, X_left3, X_left4, X_right1, X_right2, X_right3, X_right4 = get_context(X_wrds)
    X_wrds_inds = pad_sequences(X_wrds_inds, maxlen=X_max_len, dtype='int32', padding='post')
    X_left1 = pad_sequences(X_left1, maxlen=X_max_len, dtype='int32', padding='post')
    X_left2 = pad_sequences(X_left2, maxlen=X_max_len, dtype='int32', padding='post')
    X_left3 = pad_sequences(X_left3, maxlen=X_max_len, dtype='int32', padding='post')
    X_left4 = pad_sequences(X_left4, maxlen=X_max_len, dtype='int32', padding='post')
    X_right1 = pad_sequences(X_right1, maxlen=X_max_len, dtype='int32', padding='post')
    X_right2 = pad_sequences(X_right2, maxlen=X_max_len, dtype='int32', padding='post')
    X_right3 = pad_sequences(X_right3, maxlen=X_max_len, dtype='int32', padding='post')
    X_right4 = pad_sequences(X_right4, maxlen=X_max_len, dtype='int32', padding='post')
    # print(X_left1.shape)
    decoder_input = np.zeros_like(X_wrds_inds)
    decoder_input[:, 1:] = X_wrds_inds[:, :-1]
    decoder_input[:, 0] = 1
    scaler = MinMaxScaler()
    scaler.fit(X_fts)
    X_fts = scaler.transform(X_fts)
    global counter
    input_dict['input1'][counter:counter+len(X_wrds)] = X_wrds_inds
    input_dict['input2'][counter:counter+len(X_wrds)] = decoder_input
    input_dict['input3'][counter:counter+len(X_wrds)] = X_right1
    input_dict['input4'][counter:counter+len(X_wrds)] = X_right2
    input_dict['input5'][counter:counter+len(X_wrds)] = X_right3
    input_dict['input6'][counter:counter+len(X_wrds)] = X_right4
    input_dict['input7'][counter:counter+len(X_wrds)] = X_left1
    input_dict['input8'][counter:counter+len(X_wrds)] = X_left2
    input_dict['input9'][counter:counter+len(X_wrds)] = X_left3
    input_dict['input10'][counter:counter+len(X_wrds)] = X_left4
    input_dict['input11'][counter:counter+len(X_wrds)] = X_fts



def convert_sentence_to_output_format(sentence):
    X_wrds_inds = encode_words(sentence)
    X_wrds_inds = pad_sequences(X_wrds_inds, maxlen=X_max_len, dtype='int32', padding='post')
    encoded = np.array([to_categorical(X_wrds_inds[i], num_classes=Vocabulary_size) for i in range(len(X_wrds_inds))])
    return encoded

def convert_features_to_output_format(inp):
    sentence_dict={}


def train(file):
    global counter
    curr_sent_length=2
    current_sentence=[]
    for line in file.readlines():
        inp = list(line.split('\t'))
        if len(inp)==1 and inp[0]!='\n':
            inp = list(inp[0].split())
            if inp[1] == "text":
                sentence = inp[3:]
                curr_sent_length=len(sentence)
                generate_input(sentence)
        
        if len(inp)>4:
            current_sentence.append(inp[1])
            curr_sent_length-=1

        if curr_sent_length==0:
            output_dict['td2'][counter:counter+len(current_sentence)] = convert_sentence_to_output_format(current_sentence)
            counter+=len(current_sentence)
            current_sentence = []
            break


def encode_and_fill_features(file):
    ctr = 0
    for line in file.readlines():
        inp = list(line.split('\t'))
        if len(inp)>2:
            feature_dict={section.split('=')[0]:section.split('=')[1] for section in inp[5].split('|')}
            feature_dict['POS']=inp[3]
            if 'Case' not in feature_dict:
                feature_dict['Case']='Unk'
            if 'Number' not in feature_dict:
                feature_dict['Number']='Unk'
            if 'Person' not in feature_dict:
                feature_dict['Person']='Unk'
            if 'Gender' not in feature_dict:
                feature_dict['Gender']='Unk'
            if len(inp)>8:
                feature_dict2={section.split('=')[0]:section.split('=')[1] for section in inp[9].split('|')}
                if 'Tam' in feature_dict2:
                    feature_dict['TAM']=feature_dict2['Tam']
            if 'TAM' not in feature_dict:
                feature_dict['TAM']='Unk'
            output_dict['output1']=feature_dict['POS']
            output_dict['output2']=feature_dict['Gender']
            output_dict['output3']=feature_dict['Number']
            output_dict['output4']=feature_dict['Person']
            output_dict['output5']=feature_dict['Case']
            output_dict['output6']=feature_dict['TAM']


            # print(len(feature_dict))
        if ctr>20:
            break
        ctr+=1

# file = open('treebank-dev.conllu','r')
# train(file)
# file.close()
encode_and_fill_features(open('treebank-dev.conllu', 'r'))
# # for i in range(len(input_arr[0])):
# #     print(input_arr[0][i].shape)
# print (len(input_arr))
# input_file = open('input-train.pickle','wb')
# pickle.dump(input_arr, input_file)
# input_file.close()
# '''
# TODO: Find out how to generate output in required format from the given training data
# '''
# convert_sentence_to_output_format(["यह", "एशिया", "की" ,"सबसे", "बड़ी" ,"मस्जिदों" ,"में", "से", "एक","है", "।"])