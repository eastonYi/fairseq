gpu=$1
SAVE_DIR=exp/finetune_w2v_cif
W2V_PATH=../libri/wav2vec2_small.pt
DATA_DIR=data
label_type=char

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu fairseq-train $DATA_DIR \
--save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR \
--train-subset train --valid-subset valid --no-epoch-checkpoints  \
--labels $label_type --num-workers 4 --max-update 80000 \
--arch wav2vec_cif --task audio_cif --criterion qua_ce_acc --best-checkpoint-metric acc \
--w2v-path $W2V_PATH \
--assigner-conv-layers '[(512,3,1)] * 2 + [(512,2,1)] * 1' \
--apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 \
--feature-grad-mult 0.0 --freeze-finetune-updates 200 \
--validate-after-updates 1  --validate-interval 1 \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 4e-05 --lr-scheduler tri_stage \
--warmup-steps 5000 --hold-steps 20000 --decay-steps 30000 --final-lr-scale 0.05 \
--final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 \
--attention-dropout 0.0 --max-tokens 2200000 --seed 2337 --ddp-backend no_c10d --update-freq 12 \
--log-interval 20 --log-format simple --save-interval 1
