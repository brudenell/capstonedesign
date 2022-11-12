### Preprocessing Input Data
```python 
prepare_data_mecab_bpe.py --input {text-file} --output {output-file}
```
`text-file`: 인풋 텍스트 파일.
기본 포맷은 emotion_tag \t sentence \t slot_tag_labels

`output-file`: seq.in, seq.out, lable

### Albert Fine-Tuning (Intent Classification & Slot-Tagging)
```
python train_joint_bert.py --train={train_data} --save={save_model_path} --epochs=30
python instant_infer_mecab_bpe.py --model=saved_models/joint_bert_model_epochs30_excel
```

### 참고
https://github.com/MahmoudWahdan/dialog-nlu
