gpu=$1
W2V_PATH=../libri/wav2vec2_small.pt
BERT='bert-base-chinese'
SAVE_DIR=exp/finetune_w2v_cif2_bert_base
DATA_DIR=data
label_type=char

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu fairseq-train $DATA_DIR \
--save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR \
--train-subset train --valid-subset valid --no-epoch-checkpoints \
--labels $label_type --num-workers 4 --max-update 80000 \
--lambda-alpha 0.0 --lambda-qua 0.1 --lambda-embedding 0.0 \
--arch wav2vec_cif2_bert_2 --task audio_cif_bert --criterion cif_bert --best-checkpoint-metric uer \
--w2v-path $W2V_PATH --bert-name $BERT \
--apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.0 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 \
--feature-grad-mult 0.0 --freeze-finetune-updates 1000 --gold-updates 30000 --gold-rate-range '(0.9, 0.0)' \
--validate-after-updates 10000  --validate-interval 1 --save-interval 1 \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 4e-05 --lr-scheduler tri_stage \
--warmup-steps 5000 --hold-steps 30000 --decay-steps 30000 --final-lr-scale 0.05 \
--final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 \
--attention-dropout 0.0 --max-tokens 1000000 --seed 2337 --ddp-backend no_c10d --update-freq 1 \
--log-interval 500 --log-format simple
