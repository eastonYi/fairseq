gpu=$1
DATA_BIN=data
EXP_DIR=exp/bertx6_200M

TOKENIZERS_PARALLELISM=false  CUDA_VISIBLE_DEVICES=$gpu fairseq-train \
  $DATA_BIN --save-dir $EXP_DIR \
  --arch masked_lm --task masked_lm \
  --dropout 0.2 --encoder-embed-dim 512 --encoder-layers 6 --mask-prob 0.12 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 10000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 16000 --update-freq 1 --max-update 500000 --log-format simple --log-interval 2000 \
  --num-workers 4 --save-interval 1 --ddp-backend=no_c10d
