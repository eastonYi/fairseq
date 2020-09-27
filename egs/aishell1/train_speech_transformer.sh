gpu=$1
SAVE_DIR=exp/speech_transformer/
DATA_DIR=data/char

TOKENIZERS_PARALLELISM=false CUDA_LAUNCH_BLOCKING=$gpu CUDA_VISIBLE_DEVICES=$gpu fairseq-train $DATA_DIR \
--save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR \
--num-workers 4 --task speech_recognition --criterion cross_entropy --arch speech_transformer \
--train-subset train --valid-subset dev --dict dict.char.txt \
--optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr 8e-05 --lr-scheduler polynomial_decay \
--total-num-update 400000 --lr 0.0005 --warmup-updates 10000 \
--encoder-layerdrop 0.05 --max-source-positions 8000 \
--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --layernorm-embedding \
--max-tokens 40000 --max-update 400000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d \
--update-freq 1 --log-interval 20 --log-format simple --save-interval 5
