. ../path.sh

gpu=$1
lm_weight=1.6
word_score=0.4
# lm_weight=1.6
# word_score=0.4
label_type=word
# DATA_DIR=data/test_clean
# data_name=test_clean
DATA_DIR=../callhome_new/data/test_en
data_name=test
BERT='bert-base-uncased'
lexicon=data/lexicon.txt
lm=data/5-gram.word.bin
MODEL_PATH=exp/w2v_cif_bert_10h/checkpoint_best.pt
RESULT_DIR=exp/w2v_cif_bert_10h/decode_callhome_${data_name}_lm${lm_weight}_ws${word_score}

echo 'lm_weight: ' $lm_weight 'word_score: ' $word_score
CUDA_VISIBLE_DEVICES=$gpu python ../../examples/speech_recognition/infer_w2l.py $DATA_DIR \
--bert-name $BERT --task audio_cif_bert --nbest 1 --path $MODEL_PATH \
--gen-subset $data_name --results-path $RESULT_DIR --w2l-decoder kenlm \
--lm-model $lm --lm-weight ${lm_weight} \
--word-score $word_score --sil-weight 0 --criterion ctc \
--labels $label_type --max-tokens 4000000 \
--post-process $label_type --remove-bpe ' ##' --lexicon $lexicon --beam 100
