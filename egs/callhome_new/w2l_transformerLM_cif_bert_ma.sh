. ./path.sh

gpu=$1
lm_weight=0.05
word_score=0.0
label_type=char
DATA_DIR=data/test_ma
data_name=test
BERT='bert-base-chinese'
lexicon=lm/lexicon.txt
lm=lm/transformerx4_LM.pt
MODEL_PATH=exp/w2v_cif_bert_ma/checkpoint_best.pt
RESULT_DIR=exp/w2v_cif_bert_ma/decode_callhome_${data_name}_selflm${lm_weight}_score${word_score}

CUDA_VISIBLE_DEVICES=$gpu python ../../examples/speech_recognition/infer_w2l.py $DATA_DIR \
--bert-name $BERT --task audio_cif_bert --nbest 1 --path $MODEL_PATH --iscn \
--gen-subset $data_name --results-path $RESULT_DIR --w2l-decoder fairseqlm \
--lm-model $lm --lm-weight ${lm_weight} --sil-weight 0.0 \
--word-score ${word_score} --criterion ctc \
--labels $label_type --max-tokens 4000000 \
--post-process $label_type --lexicon $lexicon --beam 1
