gpu=$1
SAVE_DIR=exp/semi_pretrain_v1_20ms_100h/
DATA_DIR=data/phone

# from scratch
# field  stride 20ms
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu fairseq-train $DATA_DIR \
--save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR --num-workers 2 \
--task audio_ali --labels ali --ali-rate 2 --criterion wav2vec_v1 --arch wav2vec2_v1 \
--train-subset train_100h --valid-subset dev_clean \
--extractor-mode default --infonce \
--conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' \
--optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --total-num-update 400000 \
--lr 0.0005 --warmup-updates 20000 --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
--encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.1 \
--conv-pos 128 --conv-pos-groups 16 --loss-weights '[10]' \
--max-sample-size 300000 --min-sample-size 32000 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--max-tokens 700000 --max-update 400000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d \
--log-format simple --log-interval 200 --update-freq 8 --save-interval 1
