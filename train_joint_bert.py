# -*- coding: utf-8 -*-
"""
@author: demiust
"""

from readers.goo_format_reader import Reader
from vectorizers.bert_vectorizer import BERTVectorizer
from vectorizers.tags_vectorizer import TagsVectorizer
from models.joint_bert import JointBertModel

import argparse
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pickle
import tensorflow as tf
#
tf.compat.v1.disable_eager_execution()
#


# read command-line parameters
parser = argparse.ArgumentParser('Training the Joint BERT NLU model')
parser.add_argument('--train', '-t', help = 'Path to training data in Goo et al format', type = str, required = True)
parser.add_argument('--val', '-v', help = 'Path to validation data in Goo et al format', type = str, default = "", required = False)
parser.add_argument('--save', '-s', help = 'Folder path to save the trained model', type = str, required = True)
parser.add_argument('--epochs', '-e', help = 'Number of epochs', type = int, default = 5, required = False)
parser.add_argument('--batch', '-bs', help = 'Batch size', type = int, default = 64, required = False)
#parser.add_argument('--type', '-tp', help = 'bert   or    albert', type = str, default = 'bert', required = False)


#VALID_TYPES = ['bert', 'albert']

args = parser.parse_args()
train_data_folder_path = args.train
val_data_folder_path = args.val
save_folder_path = args.save
epochs = args.epochs
batch_size = args.batch
#type_ = args.type
print('train_data_folder_path:', train_data_folder_path)

## this line is to disable gpu
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

tf.compat.v1.random.set_random_seed(7)

'''
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=0, 
                        inter_op_parallelism_threads=0,
                        allow_soft_placement=True,
                        device_count = {'GPU': 1})
'''


config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=0, 
                        inter_op_parallelism_threads=0,
                        allow_soft_placement=True,
                        device_count = {'GPU': 1})
sess = tf.compat.v1.Session(config=config)

#if type_ == 'bert':
#    bert_model_hub_path = "./bert-module" # to use KorBert by Etri
#    is_bert = True
#elif type_ == 'albert':
#    bert_model_hub_path = './korwiki_mecab_module'
#    is_bert = False
#else:
#    raise ValueError('type must be one of these values: %s' % str(VALID_TYPES))

bert_model_hub_path = './albert-module'
is_bert = False

print('read data ...')


# validation path가 잘 설정되어 있으면 val도 같이 train
if val_data_folder_path:
    train_text_arr, train_tags_arr, train_intents = Reader.read(train_data_folder_path)
    val_text_arr, val_tags_arr, val_intents = Reader.read(val_data_folder_path)

    print('train_text_arr[0:2] :', train_text_arr[0:2])
    print('train_tags_arr[0:2] :', train_tags_arr[0:2])
    print('train_intents[0:2] :', train_intents[0:2])

    print('vectorize data ...')
    bert_vectorizer = BERTVectorizer(sess, is_bert, bert_model_hub_path) 
    # now bert model hub path exists --> already tokenized dataset
    # bert vectorizer MUST NOT tokenize input !!!

    print('bert vectorizer started ...') #valid pos removed
    train_input_ids, train_input_mask, train_segment_ids, train_sequence_lengths = bert_vectorizer.transform(train_text_arr)
    val_input_ids, val_input_mask, val_segment_ids, val_sequence_lengths = bert_vectorizer.transform(val_text_arr)

    print('vectorize tags ...')
    tags_vectorizer = TagsVectorizer()
    tags_vectorizer.fit(train_tags_arr)
    train_tags = tags_vectorizer.transform(train_tags_arr, train_input_ids)
    #val_tags = tags_vectorizer.transform(val_tags_arr, val_valid_positions)
    val_tags = tags_vectorizer.transform(val_tags_arr, val_input_ids)
    print('train_tags :', train_tags[0:2])
    slots_num = len(tags_vectorizer.label_encoder.classes_)
    print('slot num :', slots_num, tags_vectorizer.label_encoder.classes_)
    print('encode labels ...')
    intents_label_encoder = LabelEncoder()
    train_intents = intents_label_encoder.fit_transform(train_intents).astype(np.int32)
    #val_intents = intents_label_encoder.transform(val_intents).astype(np.int32)
    val_intents = intents_label_encoder.transform(val_intents).astype(np.int32)
    intents_num = len(intents_label_encoder.classes_)
    print('intents num :', intents_num)

    model = JointBertModel(slots_num, intents_num, bert_model_hub_path, sess, 
                        num_bert_fine_tune_layers=10, is_bert=is_bert)


    print('train input shape :', train_input_ids.shape, train_input_ids[0:2])
    print('train_input_mask :', train_input_mask.shape, train_input_mask[0:2])
    print('train_segment_ids :', train_segment_ids.shape, train_segment_ids[0:2])
    print('train_tags :', train_tags.shape, train_tags[0:2])
    print('train_intents :', train_intents.shape, train_intents[0:2])


    print('training model ...')

    model.fit([train_input_ids, train_input_mask, train_segment_ids], [train_tags, train_intents],
            validation_data=([val_input_ids, val_input_mask, val_segment_ids], [val_tags, val_intents]),
            epochs=epochs, batch_size=batch_size)

# validation 없을 경우 train데이터로만!
else:
    train_text_arr, train_tags_arr, train_intents = Reader.read(train_data_folder_path)
    #val_text_arr, val_tags_arr, val_intents = Reader.read(val_data_folder_path)

    print('train_text_arr[0:2] :', train_text_arr[0:2])
    print('train_tags_arr[0:2] :', train_tags_arr[0:2])
    print('train_intents[0:2] :', train_intents[0:2])

    print('vectorize data ...')
    bert_vectorizer = BERTVectorizer(sess, is_bert, bert_model_hub_path) 
    # now bert model hub path exists --> already tokenized dataset
    # bert vectorizer MUST NOT tokenize input !!!

    print('bert vectorizer started ...') #valid pos removed
    train_input_ids, train_input_mask, train_segment_ids, train_sequence_lengths = bert_vectorizer.transform(train_text_arr)
    #val_input_ids, val_input_mask, val_segment_ids, val_valid_positions, val_sequence_lengths = bert_vectorizer.transform(val_text_arr)

    print('vectorize tags ...')
    tags_vectorizer = TagsVectorizer()
    tags_vectorizer.fit(train_tags_arr)
    train_tags = tags_vectorizer.transform(train_tags_arr, train_input_ids)
    #val_tags = tags_vectorizer.transform(val_tags_arr, val_valid_positions)
    print('train_tags :', train_tags[0:2])
    slots_num = len(tags_vectorizer.label_encoder.classes_)
    print('slot num :', slots_num, tags_vectorizer.label_encoder.classes_)

    print('encode labels ...')
    intents_label_encoder = LabelEncoder()
    train_intents = intents_label_encoder.fit_transform(train_intents).astype(np.int32)
    #val_intents = intents_label_encoder.transform(val_intents).astype(np.int32)
    intents_num = len(intents_label_encoder.classes_)
    print('intents num :', intents_num)

    model = JointBertModel(slots_num, intents_num, bert_model_hub_path, sess, 
                        num_bert_fine_tune_layers=10, is_bert=is_bert)


    print('train input shape :', train_input_ids.shape, train_input_ids[0:2])
    print('train_input_mask :', train_input_mask.shape, train_input_mask[0:2])
    print('train_segment_ids :', train_segment_ids.shape, train_segment_ids[0:2])
    print('train_tags :', train_tags.shape, train_tags[0:2])
    print('train_intents :', train_intents.shape, train_intents[0:2])


    print('training model ...')

    model.fit([train_input_ids, train_input_mask, train_segment_ids], [train_tags, train_intents],
    #          validation_data=([val_input_ids, val_input_mask, val_segment_ids, val_valid_positions], [val_tags, val_intents]),
            validation_data=None, 
            epochs=epochs, batch_size=batch_size)    

### saving
print('Saving ..')
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
    print('Folder `%s` created' % save_folder_path)
model.save(save_folder_path)
with open(os.path.join(save_folder_path, 'tags_vectorizer.pkl'), 'wb') as handle:
    pickle.dump(tags_vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(save_folder_path, 'intents_label_encoder.pkl'), 'wb') as handle:
    pickle.dump(intents_label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)


tf.compat.v1.reset_default_graph()
