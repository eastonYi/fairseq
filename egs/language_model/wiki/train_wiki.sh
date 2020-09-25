gpu=$1

DATA_BIN=/data3/easton/data/TEXT/data-bin/wikitext-103
EXP_DIR=exp/transformer_wikitext-103

TOKENIZERS_PARALLELISM=false  CUDA_VISIBLE_DEVICES=$gpu fairseq-train \
  --task language_modeling $DATA_BIN --save-dir $EXP_DIR \
  --arch transformer_lm --share-decoder-input-output-embed --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 1024 --update-freq 16 --max-update 50000
