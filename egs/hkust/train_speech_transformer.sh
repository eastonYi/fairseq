gpu=$1
SAVE_DIR=exp/speech_transformer/
DATA_DIR=data2

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu fairseq-train $DATA_DIR \
--save-dir $DATA_DIR --tensorboard-logdir $SAVE_DIR \
--num-workers 8 --task speech_recognition --criterion cross_entropy_acc --arch speech_transformer \
--train-subset train --valid-subset valid \
--optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr 8e-05 --lr-scheduler polynomial_decay \
--total-num-update 400000 --lr 0.0005 --warmup-updates 32000 \
--encoder-layerdrop 0.05 --max-source-positions 1000 \
--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--max-tokens 50000 --max-update 400000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d \
--update-freq 12 --log-interval 20 --log-format simple --save-interval 1
