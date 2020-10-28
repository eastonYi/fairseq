gpu=$1
SAVE_DIR=exp/pretrain_40ms_960h_1/
DATA_DIR=data

# from scratch
# field  stride 40ms
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu fairseq-train $DATA_DIR \
--save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR --num-workers 4 \
--task audio_pretraining --criterion wav2vec --arch wav2vec2 --enable-padding \
--train-subset train_960h --valid-subset dev_clean --no-epoch-checkpoints \
--log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
--conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 3' --final-dim 256 --latent-vars 320 \
--latent-groups 2 --latent-temp '(2,0.5,0.999995)' --infonce --optimizer adam \
--adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --total-num-update 400000 \
--lr 0.0005 --warmup-updates 32000 --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
--encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.1 \
--loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 --num-negatives 100 --cross-sample-negatives 0 \
--max-sample-size 300000 --min-sample-size 32000 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--max-tokens 1000000 --max-update 400000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d \
--log-format simple --log-interval 1000 --update-freq 8 --save-interval 1
