gpu=$1
SAVE_DIR=exp/finetune_w2v_ctc_ge_subword
W2V_PATH=exp/pretrain_6lang/checkpoint_best.pt
DATA_DIR=data/ge/subword
label_type=subword

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu fairseq-train $DATA_DIR \
--save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR --post-process $label_type \
--train-subset train --valid-subset dev_ge --no-epoch-checkpoints --best-checkpoint-metric uer \
--num-workers 2 --max-update 80000 --task audio_ctc --arch wav2vec_ctc --w2v-path $W2V_PATH \
--labels $label_type --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 --zero-infinity \
--feature-grad-mult 0.0 --freeze-finetune-updates 1000 --validate-after-updates 5000  --validate-interval 5 \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 2e-05 --lr-scheduler tri_stage \
--warmup-steps 5000 --hold-steps 20000 --decay-steps 30000 --final-lr-scale 0.05 \
--final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion ctc \
--attention-dropout 0.0 --max-tokens 1000000 --seed 2337 --ddp-backend no_c10d --update-freq 1 \
--log-interval 100 --log-format simple --save-interval 10
