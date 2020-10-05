gpu=$1
SAVE_DIR=exp/finetune_w2v_seq2seq_2.2
W2V_PATH=../libri/wav2vec2_small.pt
DATA_DIR=data/ma/hkust_style_char
label_type=char

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu fairseq-train $DATA_DIR \
--save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR \
--train-subset train --valid-subset dev --no-epoch-checkpoints  \
--labels $label_type --num-workers 4 --max-update 80000 \
--arch wav2vec_seq2seq --task audio_seq2seq --criterion ls_ce_acc \
--best-checkpoint-metric acc --maximize-best-checkpoint-metric \
--w2v-path $W2V_PATH \
--decoder-embed-dim 768 --decoder-ffn-embed-dim 1536 --decoder-layers 1 --decoder-layerdrop 0.0 \
--decoder-attention-heads 2 \
--decoder-learned-pos --decoder-normalize-before \
--decoder-dropout 0.1 --decoder-attention-dropout 0.1 --decoder-activation-dropout 0.1 \
--apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 \
--feature-grad-mult 0.0 --freeze-finetune-updates 2000 \
--validate-after-updates 1  --validate-interval 1 \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 8e-05 --lr-scheduler tri_stage \
--warmup-steps 10000 --hold-steps 20000 --decay-steps 30000 --final-lr-scale 0.05 \
--final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 \
--attention-dropout 0.0 --max-tokens 800000 --seed 2337 --ddp-backend no_c10d --update-freq 1 \
--log-interval 50 --log-format simple --save-interval 1
