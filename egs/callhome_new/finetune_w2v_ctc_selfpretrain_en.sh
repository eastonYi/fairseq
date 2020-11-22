gpu=$1
SAVE_DIR=exp/finetune_w2v_ctc_self_pretrain_acc0.44_20ms_en
W2V_PATH=../libri/exp/pretrain_20ms_960h/checkpoint_best.pt
DATA_DIR=data/en/subword
label_type=subword

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu fairseq-train $DATA_DIR \
--train-subset train --valid-subset dev --criterion ctc --best-checkpoint-metric uer \
--save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR --post-process $label_type \
--num-workers 4 --max-update 80000 --task audio_ctc --arch wav2vec_ctc \
--w2v-path $W2V_PATH --labels $label_type --apply-mask --mask-selection static --mask-other 0 \
--mask-length 10 --mask-prob 0.5 --layerdrop 0.1 --mask-channel-selection static \
--mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 --zero-infinity \
--feature-grad-mult 0.0 --freeze-finetune-updates 1000 --validate-after-updates 10000 \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 \
--no-epoch-checkpoints \
--lr 4e-05 --lr-scheduler tri_stage --warmup-steps 8000 --hold-steps 42000 --decay-steps 50000 \
--final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 \
--attention-dropout 0.0 --max-tokens 1000000 --seed 2337 --ddp-backend no_c10d \
--update-freq 1 --log-interval 100 --validate-interval 1 --log-format simple --save-interval 1
