. ./path.sh

gpu=$1
lm_weight=0.2
word_score=0.4
label_type=char
DATA_DIR=data/test_ma
data_name=test
BERT='bert-base-chinese'
# lexicon=lm/lexicon.txt
# lm=lm/5-gram.bin
lexicon=/data/sxu/easton/projects/fairseq/egs/callhome_new/data/train_ma/lexicon.txt
lm=/data/sxu/easton/projects/fairseq/egs/callhome_new/data/train_ma/5-gram.word.bin
MODEL_PATH=exp/w2v_cif_bert_ma/checkpoint_best.pt
RESULT_DIR=exp/w2v_cif_bert_ma/decode_callhome_${data_name}_lm${lm_weight}

CUDA_VISIBLE_DEVICES=$gpu python ../../examples/speech_recognition/infer_w2l.py $DATA_DIR \
--bert-name $BERT --task audio_cif_bert --nbest 1 --path $MODEL_PATH --iscn \
--gen-subset $data_name --results-path $RESULT_DIR --w2l-decoder kenlm \
--lm-model $lm --lm-weight ${lm_weight} \
--word-score $word_score --sil-weight 0 --criterion ctc \
--labels $label_type --max-tokens 4000000 \
--post-process $label_type --lexicon $lexicon --beam 100
