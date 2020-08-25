DATA_DIR=./data/aishell2/preprocessed_data_vq
SAVE_DIR=./exp/vq_wav2vec_aishell2_bert_finetune_phone
LOG_FILE=./run_bert_finetune_aishell2.phone.log
#### CUDA_VISIBLE_DEVICES=0,1,2,3  nohup  python -u ./train_finetune.py $DATA_DIR --save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR --train-subset train --num-workers 4 --save-interval-updates 500 --keep-interval-updates 10 --no-epoch-checkpoints --task speech_recognition_vq --criterion ctc_loss --max-positions 6144 --arch roberta_base --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 --clip-norm 5.0 --lr-scheduler  polynomial_decay --lr 0.0001  --warmup-updates 2000 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --max-tokens 3072 --update-freq 1 --max-update 20000 --seed 5 --skip-invalid-size-inputs-valid-test --log-interval 2 --log-format simple --user-dir examples/speech_recognition/  --restore-file ./exp/vq_wav2vec_librispeech/vq_wav2vec_librispeech_bert/checkpoint_best.pt --reset-optimizer --reset-lr-scheduler --reset-dataloader --mask-prob 0.0375 --lm-alpha 0.5 --lm-beta 1.0 --lm-model-path "./data/librispeech/4-gram.bin" --lm-beam 300 --ddp-backend=no_c10d > $LOG_FILE &
CUDA_VISIBLE_DEVICES=0,1,2,3  nohup  python -u ./train_finetune.py $DATA_DIR --save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR --train-subset train --num-workers 4 --save-interval-updates 500 --keep-interval-updates 10 --no-epoch-checkpoints --task speech_recognition_vq --criterion ctc_loss --max-positions 6144 --arch roberta_base --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 --clip-norm 5.0 --lr-scheduler tri_stage --lr 0.0001 --warmup-steps 2000 --hold-steps 2000 --decay-steps 14000 --init-lr-scale 1e-5 --final-lr-scale 1e-5 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --max-tokens 3072 --update-freq 1 --max-update 20000 --seed 5 --skip-invalid-size-inputs-valid-test --log-interval 2 --log-format simple --user-dir examples/speech_recognition/  --restore-file ./exp/vq_wav2vec_librispeech/vq_wav2vec_librispeech_bert/checkpoint_best.pt --reset-optimizer --reset-lr-scheduler --reset-dataloader --mask-prob 0.0375 --lm-alpha 0.5 --lm-beta 1.0 --lm-model-path "./data/librispeech/4-gram.bin" --lm-beam 300 --ddp-backend=no_c10d > $LOG_FILE &
