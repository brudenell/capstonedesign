from flask import Blueprint, render_template, request

import subprocess
from kiwipiepy import Kiwi

kiwi = Kiwi()

bp = Blueprint('main', __name__, url_prefix='/')


@bp.route('/')
def inputString():
    return render_template('input.html')


@bp.route('/NIA', methods=('POST',))
def emotionAnalysis():
    input = request.form['content']
    output = kiwi.split_into_sents(input)
    with open("data_text/test.txt", 'a') as f:
        f.write("\n0\t")
        for i, sent in enumerate(output):
            f.write(sent.text)
            if i < len(output) - 1:
                f.write("+")
    subprocess.run(['python3','prepare_data_mecab_bpe.py','-i=data_text/test.txt','-o=data/test'])
    a = subprocess.check_output(['python3','eval_joint_bert_allsents.py','-d=data/test','-m=save_model/epoch30'])
    return a.decode("utf-8")