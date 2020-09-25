. ../path.sh

gpu=$1
label_type=subword
DATA_DIR=data
data_name=test
label_type=char
MODEL_PATH=exp/finetune_seq2seq_acc_2/checkpoint_best.pt
RESULT_DIR=exp/finetune_seq2seq_acc_2/decode_beam5

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu python $SRC_ROOT/generate.py $DATA_DIR \
    --labels $label_type --gen-subset $data_name \
    --path $MODEL_PATH \
    --results-path $RESULT_DIR \
    --task audio_seq2seq \
    --beam 5 --remove-bpe --post-process --max-tokens 4000000
