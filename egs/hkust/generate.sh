gpu=$1
DATA_DIR=data2
data_name=test
label_type=char
MODEL_PATH=exp/speech_transformer_1/checkpoint_best.pt
RESULT_DIR=exp/speech_transformer_1/decode_beam5

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu fairseq-generate $DATA_DIR \
    --gen-subset $data_name \
    --path $MODEL_PATH \
    --results-path $RESULT_DIR \
    --task speech_recognition \
    --beam 5 --remove-bpe --post-process --max-tokens 100000
