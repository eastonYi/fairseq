gpu=$1
SAVE_DIR=exp/vgg_transformer_small_100h/
DATA_DIR=/home/easton/projects/wav2vec/egs/libri/data/100h

# . ../path.sh

TOKENIZERS_PARALLELISM=false  CUDA_VISIBLE_DEVICES=$gpu fairseq-train $DATA_DIR \
--save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR \
--max-epoch 80 --task speech_recognition --arch vggtransformer_2 \
--transformer-enc-config '((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 8' \
--enc-output-dim 512 \
--transformer-dec-config '((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 6' \
--train-subset train_100h --valid-subset valid \
--clip-norm 10.0  --max-tokens 10000 --num-workers 4 \
--optimizer adadelta --lr 1.0 --adadelta-eps 1e-8 --adadelta-rho 0.95 \
--log-format simple --log-interval 20 --criterion cross_entropy_acc --update-freq 6 \
--save-interval 1
