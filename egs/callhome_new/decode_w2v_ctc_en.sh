gpu=$1
label_type=subword
DATA_DIR=data/en/subword
data_name=test
MODEL_PATH=exp/finetune_w2v_ctc_semipretrain_v1_addhkust_en/checkpoint_best.pt
RESULT_DIR=exp/finetune_w2v_ctc_semipretrain_v1_addhkust_en/decode_callhome_en_beam1

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu python ../../examples/speech_recognition/infer.py $DATA_DIR \
--task audio_ctc --nbest 1 --path $MODEL_PATH \
--gen-subset $data_name --results-path $RESULT_DIR --w2l-decoder ctc_decoder \
--criterion ctc --labels $label_type --max-tokens 4000000 \
--post-process $label_type --beam 1
