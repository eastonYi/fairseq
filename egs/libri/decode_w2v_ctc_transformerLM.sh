gpu=$1
label_type=subword
DATA_DIR=data/light/subword
data_name=test_clean
MODEL_PATH=exp/finetune_w2v_ctc_subword_10h_2/checkpoint_best.pt
RESULT_DIR=exp/finetune_w2v_ctc_subword_10h_2/decode_${data_name}_beam1

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu python ../../examples/speech_recognition/infer.py $DATA_DIR \
--task audio_ctc --nbest 1 --path $MODEL_PATH \
--gen-subset $data_name --results-path $RESULT_DIR --w2l-decoder fairseqlm \
--criterion ctc --labels $label_type --max-tokens 4000000 \
--post-process $label_type --remove-bpe '@@ ' --beam 1
