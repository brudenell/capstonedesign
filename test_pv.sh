for i in 0.6 0.7 0.8 0.9
do
    python eval_joint_bert_allsents.py -d=data/test -m=save_model/epoch30 --pv $i
done
