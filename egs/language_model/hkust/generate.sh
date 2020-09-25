gpu=$1
DATA_BIN=data-bin/hkust_char
RESULT_DIR=data-bin/hkust_char/generate
MODEL=exp/lstm_hkust_char/checkpoint_best.pt

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu \
fairseq-generate $DATA_BIN \
--task language_modeling --path $MODEL --gen-subset valid \
--skip-invalid-size-inputs-valid-test \
--max-tokens 500 --beam 1 \
--results-path $RESULT_DIR
