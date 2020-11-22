gpu=$1
SAVE_DIR=exp/semi_pretrain_v2_40ms_960h/
DATA_DIR=data/phone

# from scratch
# field  stride 40ms
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu fairseq-train $DATA_DIR \
--save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR --num-workers 4 \
--task audio_ctc --labels phone --criterion wav2vec_v2 --arch wav2vec2_v2 \
--train-subset train_960h --valid-subset dev_clean --no-epoch-checkpoints \
--log-keys '["prob_perplexity","code_perplexity","temp"]' --extractor-mode default \
--conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 3' \
--optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --total-num-update 400000 \
--lr 0.0005 --warmup-updates 32000 --mask-length 10 --mask-prob 0.4 --mask-selection static --mask-other 0 \
--encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.1 \
--loss-weights '[10]' --conv-pos 128 --conv-pos-groups 16 \
--max-sample-size 300000 --min-sample-size 32000 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--max-tokens 1400000 --max-update 400000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d \
--log-format simple --log-interval 100 --update-freq 8 --save-interval 1
