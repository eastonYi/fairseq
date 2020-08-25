SAVE_DIR=exp/wav2vec2_base_finetune_callhome
W2V_PATH=exp/wav2vec2_base_pretrain/checkpoint_best.pt
DATA_DIR=data/callhome/train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py $DATA_DIR --save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR --fp16 \
--wer-args '("data/callhome/train/callhome.arpa.bin","data/callhome/train/lexicon.txt",2,-1)' \
--post-process letter --train-subset train --valid-subset "valid" --no-epoch-checkpoints --best-checkpoint-metric uer --num-workers 4 \
--max-update 80000 --sentence-avg --task audio_pretraining --arch wav2vec_ctc --w2v-path $W2V_PATH \
--labels ltr --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 --zero-infinity \
--feature-grad-mult 0.0 --freeze-finetune-updates 10000 --validate-after-updates 10000  --validate-interval 50 --optimizer adam \
--adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 2e-05 --lr-scheduler tri_stage --warmup-steps 8000 --hold-steps 42000 \
--decay-steps 50000 --final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion ctc \
--attention-dropout 0.0 --max-tokens 1280000 --seed 2337 --ddp-backend no_c10d --update-freq 3 \
--log-interval 10 --log-format simple --save-interval 1
