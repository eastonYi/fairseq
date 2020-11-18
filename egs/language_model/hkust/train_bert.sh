gpu=$1
DATA_BIN=data/hkust_char_add_callhome_ma
EXP_DIR=exp/bertx2_hkust_char_add_callhome_ma

TOKENIZERS_PARALLELISM=false  CUDA_VISIBLE_DEVICES=$gpu fairseq-train \
  $DATA_BIN --save-dir $EXP_DIR \
  --arch masked_lm --task masked_lm \
  --dropout 0.1 --encoder-embed-dim 512 --encoder-layers 2 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 1000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 14000 --update-freq 4 --max-update 10000 --log-format simple --log-interval 20 \
  --num-workers 4 --save-interval 10
