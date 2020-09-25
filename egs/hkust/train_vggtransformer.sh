gpu=$1
SAVE_DIR=exp/vgg_transformer/
DATA_DIR=data2

. ../path.sh

TOKENIZERS_PARALLELISM=false  CUDA_VISIBLE_DEVICES=$gpu python $SRC_ROOT/train.py $DATA_DIR \
--save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR \
--max-epoch 80 --task speech_recognition --arch vggtransformer_2 \
--train-subset train --valid-subset valid \
--clip-norm 10.0  --max-tokens 5000 \
--optimizer adadelta --lr 1.0 --adadelta-eps 1e-8 --adadelta-rho 0.95 \
--log-format simple --log-interval 20 --criterion cross_entropy_acc --update-freq 6 \
--save-interval 1
