. ../path.sh

gpu=$1
label_type=char
DATA_DIR=data2
data_name=test
label_type=char
MODEL_PATH=exp/speech_transformer_2/checkpoint_best.pt
RESULT_DIR=exp/speech_transformer_2/decode_seq2seq_beam5

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu \
python $MAIN_ROOT/examples/speech_recognition/infer.py $DATA_DIR \
    --gen-subset $data_name \
    --path $MODEL_PATH \
    --results-path $RESULT_DIR \
    --task speech_recognition --w2l-decoder seq2seq_decoder \
    --criterion cross_entropy_acc  --iscn \
    --beam 5 --remove-bpe $label_type --max-tokens 10000
