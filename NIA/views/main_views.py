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
    a = subprocess.check_output(['python3','eval_joint_bert_allsents.py','-d=data/test','-m=save_model/epoch30'])
    a2 = a.decode("utf-8")
    a3 = a2.split(':')
    a4 = a3[-1].split()
    label = int(a4[0])
    score = float(a4[1])
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
              "score" : score}
    return jsonify(result)