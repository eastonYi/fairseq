. ../path.sh

gpu=$1
label_type=char
beam=1
DATA_DIR=data/char
data_name=dev
MODEL_PATH=exp/finetune_w2v_seq2seq/checkpoint_last.pt
RESULT_DIR=exp/finetune_w2v_seq2seq/decode_seq2seq_beam${beam}_${data_name}

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu \
python $MAIN_ROOT/examples/speech_recognition/infer.py $DATA_DIR \
    --gen-subset $data_name \
    --path $MODEL_PATH \
    --labels $label_type \
    --results-path $RESULT_DIR \
    --task audio_seq2seq --w2l-decoder seq2seq_decoder \
    --criterion cross_entropy_acc --iscn \
    --beam ${beam} --remove-bpe $label_type --max-tokens 600000
