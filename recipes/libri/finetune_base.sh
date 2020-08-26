SAVE_DIR=./exp/libri/base_finetune
DATA_DIR=/data3/easton/data/fairseq/libri/finetune
MODEL_PATH=/data3/easton/data/fairseq/checkpoint_best.pt

CUDA_VISIBLE_DEVICES=0,1 python train.py $DATA_DIR --save-dir $SAVE_DIR --fp16 \
--post-process letter --valid-subset "dev_other" --no-epoch-checkpoints --best-checkpoint-metric uer --num-workers 4 \
--max-update 80000 --sentence-avg --task audio_pretraining --arch wav2vec_ctc --w2v-path $MODEL_PATH \
--labels ltr --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 --zero-infinity \
--feature-grad-mult 0.0 --freeze-finetune-updates 10000 --validate-after-updates 10000 --optimizer adam \
--adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 2e-05 --lr-scheduler tri_stage --warmup-steps 8000 --hold-steps 32000 \
--decay-steps 40000 --final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion ctc \
--attention-dropout 0.0 --max-tokens 1280000 --fix-batches-to-gpus --seed 2337 --log-format json --log-interval 500 \
--ddp-backend no_c10d --update-freq 3 --log-interval 10 --log-format simple --reset-optimizer --data-buffer-size 0
