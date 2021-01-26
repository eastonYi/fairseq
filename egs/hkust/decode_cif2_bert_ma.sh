gpu=$1
label_type=word
DATA_DIR=data/bert_style
BERT='bert-base-chinese'
data_name=test
# MODEL_PATH='exp/finetune_w2v_cif_bert_4/checkpoint_last.pt:exp/finetune_w2v_cif_bert_3/checkpoint_last.pt'
MODEL_PATH=exp/finetune_w2v_cif_bert_4/checkpoint_last.pt
RESULT_DIR=exp/finetune_w2v_cif_bert_4/decode_beam1

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu \
python ../../examples/speech_recognition/infer_cif_bert.py $DATA_DIR \
--task audio_cif_bert --nbest 1 --path $MODEL_PATH \
--gen-subset $data_name --results-path $RESULT_DIR \
--criterion ctc_cif_bert --labels $label_type --iscn --max-tokens 4000000 \
--post-process $label_type --bert-name $BERT
