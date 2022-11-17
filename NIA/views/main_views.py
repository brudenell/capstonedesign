import sys
sys.path.append("/mnt/c/flask_capstone")

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
                tags_vectorizer, intents_label_encoder, model):
    
    #inferred_tags, first_inferred_intent, first_inferred_intent_score, _, _, slots_score = model.predict_slots_intent([data_input_ids, data_input_mask, data_segment_ids], tags_vectorizer, intents_label_encoder)
    
    # for all sents
    #with graph.as_default():
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

    return int(first_inferred_intent_final[0][0]), first_inferred_intent_score_final[0]

VALID_TYPES = ['bert', 'albert']

load_folder_path = "save_model/epoch30"
data_folder_path = "data/test"
# positive_value = args.pv
#type_ = args.type

# this line is to disable gpu
os.environ['CUDA_VISIBLE_DEVICES']='-1'

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8,
                        inter_op_parallelism_threads=0,
                        allow_soft_placement=True,
                        device_count = {'CPU': 1})
#graph = tf.compat.v1.get_default_graph()


#tf.compat.v1.reset_default_graph()
#print("======= Done =======")
# endtime = time.time()
# print(endtime - starttime)



from flask import Blueprint, render_template, request, jsonify

import subprocess
from kiwipiepy import Kiwi
from konlpy.tag import Okt

okt = Okt()
kiwi = Kiwi()

bp = Blueprint('main', __name__, url_prefix='/')


@bp.route('/')
def inputString():
    return render_template('input.html')


@bp.route('/NIA', methods=('POST',))
def emotionAnalysis():
    input = request.form['content']
    input_normal = okt.normalize(input)
    output = kiwi.split_into_sents(input_normal)
    with open("data_text/test.txt", 'w') as f:
        f.write("0\t")
        for i, sent in enumerate(output):
            f.write(sent.text)
            if i < len(output) - 1:
                f.write("+")
    subprocess.run(['python3','prepare_data_mecab_bpe.py','-i=data_text/test.txt','-o=data/test'])
    #a = subprocess.check_output(['python3','eval_joint_bert_allsents.py','-d=data/test','-m=save_model/epoch30'])
    # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    # print(sess)
    # model = JointBertModel.load(load_folder_path, sess)
    sess = tf.compat.v1.Session(config=config)

    bert_model_hub_path = './albert-module'
    is_bert = False
   #tokenizer = albert_tokenization.FullTokenizer('./albert-module/assets/v0.vocab')

    bert_vectorizer = BERTVectorizer(sess, is_bert, bert_model_hub_path)

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



    #starttime = time.time()
    #print('==== Evaluation ====')
    label, score = get_results( data_input_ids, 
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
                                intents_label_encoder,
                                model)

    # a2 = a.decode("utf-8")
    # a3 = a2.split(':')
    # a4 = a3[-1].split()
    # label = int(a4[0])
    # score = float(a4[1])
    emotion = "분노"
    if label == 1:
        emotion = "슬픔"
    elif label == 2:
        emotion = "불안"
    elif label == 3:
        emotion = "상처"
    elif label == 4:
        emotion = "당황"
    elif label == 5:
        emotion = "기쁨"
    result = {"emotion" : emotion,
              "label" : label,
              "score" : float(score)}
    return jsonify(result)