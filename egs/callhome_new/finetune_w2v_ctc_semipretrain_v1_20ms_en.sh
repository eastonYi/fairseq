gpu=$1
# SAVE_DIR=exp/finetune_w2v_ctc_semipretrain_v1_addhkust_en_2
# W2V_PATH=../hkust/exp/semi_pretrain_v1/checkpoint_last.pt
SAVE_DIR=exp/finetune_w2v_ctc_semipretrain_v1_en_4
W2V_PATH=../libri/wav2vec2_semi_pretrain_v1_20ms_960h_7.pt
DATA_DIR=data/en/subword
label_type=subword

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu fairseq-train $DATA_DIR \
--train-subset train --valid-subset dev --criterion ctc --best-checkpoint-metric uer \
--save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR --post-process $label_type \
--num-workers 4 --max-update 80000 --task audio_ctc --arch wav2vec_ctc \
--w2v-path $W2V_PATH --labels $label_type --apply-mask --mask-selection static --mask-other 0 \
--mask-length 10 --mask-prob 0.5 --layerdrop 0.1 --mask-channel-selection static \
--mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 --zero-infinity \
--feature-grad-mult 0.0 --freeze-finetune-updates 1000 --validate-after-updates 5000 \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 \
--no-epoch-checkpoints \
--lr 2e-05 --lr-scheduler tri_stage --warmup-steps 8000 --hold-steps 42000 --decay-steps 50000 \
--final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 \
--attention-dropout 0.0 --max-tokens 1400000 --seed 2337 --ddp-backend no_c10d \
--update-freq 1 --log-interval 100 --validate-interval 1 --log-format simple --save-interval 1
