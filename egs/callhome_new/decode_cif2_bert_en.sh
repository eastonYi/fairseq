gpu=$1
label_type=word
DATA_DIR=data/en/bert_style
BERT='bert-base-uncased'
data_name=test
MODEL_PATH=exp/finetune_w2v_ctc_cif2_bert_base_en_2/checkpoint_best.pt
RESULT_DIR=exp/finetune_w2v_ctc_cif2_bert_base_en_2/decode_callhome_ma_beam1

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu \
python ../../examples/speech_recognition/infer_cif_bert.py $DATA_DIR \
--task audio_cif_bert --path $MODEL_PATH --bert-name $BERT \
--gen-subset $data_name --results-path $RESULT_DIR \
--criterion ctc_cif_bert --labels $label_type --max-tokens 4000000 --infer-threshold 0.8
