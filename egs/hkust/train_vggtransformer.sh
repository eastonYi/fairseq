gpu=$1
SAVE_DIR=exp/vgg_transformer/
DATA_DIR=data

TOKENIZERS_PARALLELISM=false  CUDA_VISIBLE_DEVICES=$gpu fairseq-train $DATA_DIR \
--save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR \
--max-epoch 80 --task speech_recognition --arch vggtransformer_2 \
--train-subset train --valid-subset dev --dict dict.char.txt \
--clip-norm 10.0  --max-tokens 5000 \
--optimizer adadelta --lr 1.0 --adadelta-eps 1e-8 --adadelta-rho 0.95 \
--log-format simple --log-interval 1000 --criterion ce_acc --update-freq 1 \
--save-interval 1 --ddp-backend no_c10d
