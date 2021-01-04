. ./path.sh

gpu=$1
lm_weight=0.0
word_score=0.1
label_type=word
DATA_DIR=data/test_en
data_name=test
BERT='bert-base-uncased'
lexicon=/data/sxu/easton/projects/fairseq/egs/callhome_new/data/train_en/lexicon.txt
lm=/data/sxu/easton/projects/fairseq/egs/callhome_new/data/train_en/5-gram.word.bin
MODEL_PATH=exp/w2v_cif_bert_en/checkpoint_best.pt
RESULT_DIR=exp/w2v_cif_bert_en/decode_callhome_${data_name}_lm${lm_weight}_ws${word_score}

CUDA_VISIBLE_DEVICES=$gpu python ../../examples/speech_recognition/infer_w2l.py $DATA_DIR \
--bert-name $BERT --task audio_cif_bert --nbest 1 --path $MODEL_PATH \
--gen-subset $data_name --results-path $RESULT_DIR --w2l-decoder kenlm \
--lm-model $lm --lm-weight ${lm_weight} \
--word-score $word_score --sil-weight 0 --criterion ctc \
--labels $label_type --max-tokens 4000000 \
--post-process $label_type --remove-bpe ' ##' --lexicon $lexicon --beam 100
