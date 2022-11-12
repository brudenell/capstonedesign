# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

import os
import pdb

class Reader:

    def __init__(self):
        pass

    def read(dataset_folder_path):
        labels = None
        text_arr = None
        tags_arr = None

        # intention
        with open(os.path.join(dataset_folder_path, 'label'), encoding='utf-8') as f:
            labels = f.readlines()

        # input sent
        with open(os.path.join(dataset_folder_path, 'seq.in'), encoding='utf-8') as f:
            text_arr = f.readlines()
        new_text_arr = []
        for text in text_arr:
            new_text_arr.append(text.split(' + ')[0] + '\n')

        # tags array

        with open(os.path.join(dataset_folder_path, 'seq.out'), encoding='utf-8') as f:
            tags_arr = f.readlines()
        new_tags_arr = []
        for tag in tags_arr:
            new_tags_arr.append(tag.split(' + ')[0] + '\n')

        assert len(new_text_arr) == len(new_tags_arr) == len(labels)

        #pdb.set_trace()
        return new_text_arr, new_tags_arr, labels

    def read_allsents(dataset_folder_path):
        labels = None
        text_arr = None
        tags_arr = None

        # intention
        with open(os.path.join(dataset_folder_path, 'label'), encoding='utf-8') as f:
            labels = f.readlines()

        # input sent
        with open(os.path.join(dataset_folder_path, 'seq.in'), encoding='utf-8') as f:
            text_arr = f.readlines()
        
        # tags array

        with open(os.path.join(dataset_folder_path, 'seq.out'), encoding='utf-8') as f:
            tags_arr = f.readlines()

        assert len(text_arr) == len(tags_arr) == len(labels)

        #pdb.set_trace()
        return text_arr, tags_arr, labels


if __name__ == '__main__':
    text_arr, tags_arr, labels = Reader.read_allsents('/home/tta/data1/NIA/data/train')
