
# -*- coding: utf-8 -*-
"""
@author: jungwonchang
"""

import os, re
import argparse
from mecab import MeCab
from vectorizers.albert_tokenization import FullTokenizer

import pdb

# 상황에 따라서 다른 tokenizer를 사용해도 된다.
tokenizer = FullTokenizer(vocab_file="./albert-module/assets/v0.vocab")

# "/인물;한지민/과 /인물;한예슬/ 나오는 드라마 있어?"와 같은 예시처럼
# 해당 데이터에서는 "/슬롯(레이블)명;엔티티/"의 형식으로 슬롯과 엔티티를 정리해 놨으므로,
# 이를 잡아 줄 수 있는 정규표현식을 준비한다.
slot_pattern = re.compile(r"/(.+?);(.+?)/")
multi_spaces = re.compile(r"\s+")


mecab = MeCab()
def do_mecab(string):
    return ' '.join(mecab.morphs(string))

def process_file(file_path, output_dir):
    """
    블루웨일 단방향 데이터가 있는 file_path을 argument로 주면 가공을 한 이후에
    output_dir 아래에 3개의 파일(seq.in, seq.out, lable)을 저장해 주는 함수.
    output_dir의 경우 만약 존재하지 않는다면
    """
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    data = open(file_path).read().splitlines()

    # line별로 process를 해준 뒤,
    processed_data = [process_line(line, tokenizer) for line in data]

    intentions = list(map(lambda x: x[0], processed_data))
    tokens = list(map(lambda x: x[1], processed_data))
    labels = list(map(lambda x: x[2], processed_data))

    # seq_in : 토큰들로만 이루어진 파일
    # seq_out : 레이블로들만 이루어진 파일
    intention_file = os.path.join(output_dir, "label")
    seq_in = os.path.join(output_dir, "seq.in")
    seq_out = os.path.join(output_dir, "seq.out")

    with open(intention_file, "w") as f:
        f.write("\n".join(intentions) + "\n")

    with open(seq_in, "w") as f:
        f.write("\n".join(tokens)+ "\n")

    with open(seq_out, "w") as f:
        f.write("\n".join(labels)+ "\n")


def process_line(line, tokenizer):
    """
    블루웨일 데이터를 라인별로 처리해 주는 함수이다.
    라인을 주게 되면, (토큰, 슬롯(레이블))

    (다만 토크나이저에 따라 토크나이징을 하는 방식이 상이하므로 이 부분에 대해서는 코드 수정을 해주어야 한다.)

    예를 들어 "/인물;한지민/과 /인물;한예슬/ 나오는 드라마 있어?" 같은 input을 받게 되면,
        ('한 지민 과 한예 슬 나오 는 드라마 있 어 ?', '인물 인물 O 인물 인물 O O O O O O')와 같은 (토큰, slot)쌍으로 된 결과값을 반환한다.
    """
    intention, sentence = line.split("\t")
    
    # 20201216
    # 
    sentence_list = sentence.split("+")
    sentence_tokenized_list = []
    labels_list = []

    for sentence in sentence_list:
        # slots = slot_pattern.findall(sentence)
        line_refined = slot_pattern.sub("/슬롯/", sentence)
        tokens = ""
        labels = ""
        slot_index = 0

        for word in line_refined.split():
            # "/"로 시작하면 슬롯임을 알려주므로, 우선 레이블과 엔티티를 분리해 주자.
            # "/게임명;일곱개의 대죄/" --> ("게임명", "일곱개의 대죄")
            # if word.startswith("/"):
            #     # pdb.set_trace()
            #     label, entity = slots[slot_index]
            #     slot_index += 1

            #     # 엔티티를 토크나이즈 한 후, 토큰별로 레이블을 추가해 준다.
            #     entity_tokens = " ".join(tokenizer.tokenize(do_mecab(entity)))

            #     tokens += entity_tokens + " "
            #     labels += (label + " ") * len(entity_tokens.split())

            #     # 해당 단어가 "/"로 끝나면 해당 어절은 슬롯만 포함하며, 추가적으로 해줄 사안이 없지만,
            #     # 그렇지 않다면 조사가 붙은 것이며(eg. "/슬롯/이", "/슬롯/에서"),
            #     # 조사에 대해서 추가적으로 토큰 및 레이블을 추가해 준다.
            #     if not word.endswith("/"):
            #         # 우선 "/" 뒤에 오는 조사를 찾아 주고(2글자 이상의 조사도 있기에 [-1]을 사용하지 않는다.)
            #         josa = word[word.rfind("/")+1:]
            #         josa_tokens = " ".join(tokenizer.tokenize(do_mecab(josa)))

            #         tokens += josa_tokens + " "
            #         labels += ("O" + " ") * len(josa_tokens.split())

            # # "/"로 시작하지 않는 경우는 레이블이 없는 그냥 단어이다.
            # # 따라서 토크나이즈를 해준 뒤, 그 길이만큼 "O" 레이블을 레이블에 추가해준다.
            # else:
            word_tokens = " ".join(tokenizer.tokenize(do_mecab(word)))

            tokens += word_tokens + " "
            labels += ("O" + " ") * len(word_tokens.split())

        tokens = tokens.replace('##', '')
        tokens = multi_spaces.sub(" ", tokens.strip())
        labels = multi_spaces.sub(" ", labels.strip())

        # 만일 토큰의 개수와 슬롯의 개수가 맞지 않다면 본래 라인과 더불어 토큰/슬롯들을 프린트해준다.
        if len(tokens.split()) != len(labels.split()):
            print(line)
            print("\t" + tokens + "\t", len(tokens.split()))
            print("\t" + labels + "\t", len(labels.split()))

        sentence_tokenized_list.append(tokens)
        labels_list.append(labels)
    
    joined_sents = ' + '.join(sentence_tokenized_list)
    joined_labels = ' + '.join(labels_list)
    ### labels 는 전체 length만 맞춰서 만들기

    return intention, joined_sents, joined_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help = '블루웨일 단방향 데이터 형식의 텍스트 파일', type = str, required = True)
    parser.add_argument('--output', '-o', help = 'Path to data in Goo et al format', type = str, required = True)

    args = parser.parse_args()
    file_path = args.input
    output_dir = args.output

    process_file(file_path, output_dir)
