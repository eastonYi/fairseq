gpu=$1
SAVE_DIR=exp/speech_transformer/
DATA_DIR=data

CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA_DIR \
--save-dir $DATA_DIR --num-workers 4 --task audio_seq2seq --criterion cross_entropy_acc --arch speech_transformer \
--optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay \
--total-num-update 400000 --lr 0.0005 --warmup-updates 32000 \
--encoder-layerdrop 0.05 \
--max-sample-size 250000 --min-sample-size 32000 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--max-tokens 1400000 --max-update 400000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d
