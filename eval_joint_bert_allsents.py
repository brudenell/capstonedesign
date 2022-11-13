# -*- coding: utf-8 -*-
"""
@author: jungwonchang
"""
#intent_correct 자세히 보기 first_inferred_intent_final[i], intents[i]
from readers.goo_format_reader import Reader
from vectorizers.bert_vectorizer import BERTVectorizer
from models.joint_bert import JointBertModel
from utils import flatten
from vectorizers.tags_vectorizer import TagsVectorizer
from vectorizers import albert_tokenization
import numpy as np 
import json 

import argparse
import os
import pickle
import tensorflow as tf
from sklearn import metrics

import pdb
from collections import defaultdict

import time


def get_results(input_ids, input_mask, segment_ids,  sequence_lengths, tags_arr, intents, 
                input_ids_f, input_mask_f, segment_ids_f,  sequence_lengths_f, tags_arr_f, intents_f, 
                tags_vectorizer, intents_label_encoder):
    
    #inferred_tags, first_inferred_intent, first_inferred_intent_score, _, _, slots_score = model.predict_slots_intent([data_input_ids, data_input_mask, data_segment_ids], tags_vectorizer, intents_label_encoder)
    
    # for all sents
    _, first_inferred_intent, first_inferred_intent_score, _, _, slots_score = model.predict_slots_intent([input_ids, input_mask, segment_ids], tags_vectorizer, intents_label_encoder)
    _, first_inferred_intent_f, first_inferred_intent_score_f, _, _, slots_score_f = model.predict_slots_intent([input_ids_f, input_mask_f, segment_ids_f], tags_vectorizer, intents_label_encoder)
    
    first_inferred_intent_final = []
    first_inferred_intent_score_final = []
    
    for i in range(len(first_inferred_intent)):
        # if two intentions are same:
        # if first_inferred_intent[i] == first_inferred_intent_f[i]:
        #     first_inferred_intent_final.append(first_inferred_intent[i])
        #     # adjust score
        #     #first_inferred_intent_score_final.append(first_inferred_intent_score_f[i] * 0.5 + first_inferred_intent_score[i] * 0.5)

        #     first_inferred_intent_score_final.append()

        if first_inferred_intent_score[i] > first_inferred_intent_score_f[i]:
            first_inferred_intent_final.append(first_inferred_intent[i])
            first_inferred_intent_score_final.append(first_inferred_intent_score[i])

        else:
            first_inferred_intent_final.append(first_inferred_intent_f[i])
            first_inferred_intent_score_final.append(first_inferred_intent_score_f[i])

    acc = metrics.accuracy_score(intents, first_inferred_intent_final)
    # tag_incorrect = ''
    intent_incorrect = ''
    intent_correct = ''

    for i, sent in enumerate(input_ids):
        if intents[i] != first_inferred_intent_final[i]:
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
            intent_incorrect += ('sent {}\n'.format(tokens))
            intent_incorrect += ('pred: {}\n'.format(first_inferred_intent_final[i].strip()))
            intent_incorrect += ('score: {}\n'.format(first_inferred_intent_score_final[i]))
            intent_incorrect += ('ansr: {}\n'.format(intents[i].strip()))

        else:
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
            intent_correct += ('sent {}\n'.format(tokens))
            intent_correct += ('pred: {}\n'.format(first_inferred_intent_final[i].strip()))
            intent_correct += ('score: {}\n'.format(first_inferred_intent_score_final[i]))
            intent_correct += ('ansr: {}\n'.format(intents[i].strip()))

    # f1_score
    global positive_value
    positive_value = 0.5
    pv = positive_value
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    tp_sents = ''
    tn_sents = ''
    fp_sents = ''
    fn_sents = ''

    for i in range(len(intents)):
        #
        #print("예상한 레이블:",first_inferred_intent_final[i],"    실제레이블:",intents[i],sep=" ")
        print("감정 분석 결과: ", first_inferred_intent_final[i], first_inferred_intent_score_final[i])
        #
        if first_inferred_intent_final[i] == intents[i] and first_inferred_intent_score_final[i] >= pv:
                tp += 1
                tp_sents += ('sent {}\n'.format(tokens))
                tp_sents += ('pred: {}\n'.format(first_inferred_intent_final[i].strip()))
                tp_sents += ('score: {}\n'.format(first_inferred_intent_score_final[i]))
                tp_sents += ('ansr: {}\n'.format(intents[i].strip()))

        elif first_inferred_intent_final[i] != intents[i] and first_inferred_intent_score_final[i] >= pv:
                fp += 1
                fp_sents += ('sent {}\n'.format(tokens))
                fp_sents += ('pred: {}\n'.format(first_inferred_intent_final[i].strip()))
                fp_sents += ('score: {}\n'.format(first_inferred_intent_score_final[i]))
                fp_sents += ('ansr: {}\n'.format(intents[i].strip()))

        elif first_inferred_intent_final[i] == intents[i] and first_inferred_intent_score_final[i] < pv:
                fn += 1
                fn_sents += ('sent {}\n'.format(tokens))
                fn_sents += ('pred: {}\n'.format(first_inferred_intent_final[i].strip()))
                fn_sents += ('score: {}\n'.format(first_inferred_intent_score_final[i]))
                fn_sents += ('ansr: {}\n'.format(intents[i].strip()))

        elif first_inferred_intent_final[i] != intents[i] and first_inferred_intent_score_final[i] < pv:
                tn += 1
                tn_sents += ('sent {}\n'.format(tokens))
                tn_sents += ('pred: {}\n'.format(first_inferred_intent_final[i].strip()))
                tn_sents += ('score: {}\n'.format(first_inferred_intent_score_final[i]))
                tn_sents += ('ansr: {}\n'.format(intents[i].strip()))

    #precision = tp / (tp + fp)
    #recall = tp / (tp + fn)
    #f1_score = 2 * (precision * recall) / (precision + recall)    
    #f1_score = round(f1_score, 3)
    #precision = round(precision, 3)
    #recall = round(recall, 3)
    precision = 0
    recall = 0
    f1_score = 0

    return f1_score, precision, recall, acc, intent_incorrect, intent_correct, tp, tn, fp, fn, tp_sents, tn_sents, fp_sents, fn_sents

# read command-line parameters
parser = argparse.ArgumentParser('Evaluating the Joint BERT / ALBERT NLU model')
parser.add_argument('--model', '-m', help = 'Path to joint BERT / ALBERT NLU model', type = str, required = True)
parser.add_argument('--data', '-d', help = 'Path to data in Goo et al format', type = str, required = True)
# parser.add_argument('--pv', help = 'threshold value for F1', type = float, required = True)
#parser.add_argument('--type', '-tp', help = 'bert   or    albert', type = str, default = 'bert', required = False)


VALID_TYPES = ['bert', 'albert']

args = parser.parse_args()
load_folder_path = args.model
data_folder_path = args.data
# positive_value = args.pv
#type_ = args.type

# this line is to disable gpu
os.environ['CUDA_VISIBLE_DEVICES']='-1'

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8,
                        inter_op_parallelism_threads=0,
                        allow_soft_placement=True,
                        device_count = {'GPU': 1})
sess = tf.compat.v1.Session(config=config)

bert_model_hub_path = './albert-module'
is_bert = False
tokenizer = albert_tokenization.FullTokenizer('./albert-module/assets/v0.vocab')

bert_vectorizer = BERTVectorizer(sess, is_bert, bert_model_hub_path)

# loading models
#print('Loading models ...')
if not os.path.exists(load_folder_path):
    pass
    #print('Folder `%s` not exist' % load_folder_path)

with open(os.path.join(load_folder_path, 'tags_vectorizer.pkl'), 'rb') as handle:
    tags_vectorizer = pickle.load(handle)
    slots_num = len(tags_vectorizer.label_encoder.classes_)
with open(os.path.join(load_folder_path, 'intents_label_encoder.pkl'), 'rb') as handle:
    intents_label_encoder = pickle.load(handle)
    intents_num = len(intents_label_encoder.classes_)

model = JointBertModel.load(load_folder_path, sess)

# read all sents
data_text_arr, data_tags_arr, data_intents = Reader.read_allsents(data_folder_path)
data_input_ids, data_input_mask, data_segment_ids, data_sequence_lengths = bert_vectorizer.transform(data_text_arr)

# read first sent
data_text_arr_f, data_tags_arr_f, data_intents_f = Reader.read(data_folder_path)
data_input_ids_f, data_input_mask_f, data_segment_ids_f, data_sequence_lengths_f = bert_vectorizer.transform(data_text_arr_f)


tags_vectorizer = TagsVectorizer()
# all sents + first sent
tags_vectorizer.fit(data_tags_arr + data_tags_arr_f)
data_tags_arr = tags_vectorizer.transform(data_tags_arr, data_input_ids)
data_tags_arr_f = tags_vectorizer.transform(data_tags_arr_f, data_input_ids_f)

#원래 주석
#print(data_tags_arr[1])
#print(data_input_ids[1])



starttime = time.time()
#print('==== Evaluation ====')
f1_score, precision, recall, acc, intent_incorrect, intent_correct, tp, tn, fp, fn, tp_sents, tn_sents, fp_sents, fn_sents = get_results(
                                                            data_input_ids, 
                                                            data_input_mask, 
                                                            data_segment_ids,
                                                            data_sequence_lengths,
                                                            data_tags_arr, 
                                                            data_intents,
                                                            data_input_ids_f, 
                                                            data_input_mask_f, 
                                                            data_segment_ids_f,
                                                            data_sequence_lengths_f,
                                                            data_tags_arr_f, 
                                                            data_intents_f,
                                                            tags_vectorizer, 
                                                            intents_label_encoder)

# 테스트 결과를 모델 디렉토리의 하위 디렉토리 'test_results'에 저장해 준다.
#print("saving test_results to " + load_folder_path)
#원래 주석
result_path = os.path.join(load_folder_path, 'test_results/before')
em_result_path = os.path.join(result_path, 'EM')
f1_result_path = os.path.join(result_path, 'F1')

if not os.path.isdir(result_path):
    os.mkdir(result_path)
if not os.path.isdir(em_result_path):
    os.mkdir(em_result_path)
if not os.path.isdir(f1_result_path):
    os.mkdir(f1_result_path)

## em related
with open(os.path.join(em_result_path, f'emotion_incorrect.txt'), 'w') as f:
    f.write(intent_incorrect)
with open(os.path.join(em_result_path, f'emotion_correct.txt'), 'w') as f:
    f.write(intent_correct)

## f1 related
with open(os.path.join(f1_result_path, f'true_positive.txt'), 'w') as f:
    f.write(tp_sents)
with open(os.path.join(f1_result_path, f'true_negative.txt'), 'w') as f:
    f.write(tn_sents)
with open(os.path.join(f1_result_path, f'false_positive.txt'), 'w') as f:
    f.write(fp_sents)
with open(os.path.join(f1_result_path, f'false_negative.txt'), 'w') as f:
    f.write(fn_sents)

with open(os.path.join(result_path, f'test_total.txt'), 'w') as f:
    f.write(f'''Positive value = {positive_value}
Intent f1_score = {f1_score}
Intent precision = {precision}
Intent recall = {recall}
Intent accuracy = {acc}
True Positive = {tp}
True Negative = {tn}
False Positive = {fp}
False Negative = {fn}
''')

tf.compat.v1.reset_default_graph()
#print("======= Done =======")
endtime = time.time()
print(endtime - starttime)
