lm_weight=1.46
word_score=-0.59
RESULT_DIR=exp/wav2vec2_base_finetune_callhome_4lang/decoder_result_callhome_ar_selflm${lm_weight}_score${word_score}
CUDA_VISIBLE_DEVICES=1 python examples/speech_recognition/infer.py data/callhome/test_4lang --task audio_pretraining \
--nbest 1 --path exp/wav2vec2_base_finetune_callhome_4lang/checkpoint_best.pt --gen-subset "test_ar" --results-path $RESULT_DIR --w2l-decoder kenlm \
--lm-model data/callhome/train_4lang/callhome_4lang.arpa.bin --lm-weight ${lm_weight} --word-score ${word_score} --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
--post-process letter  --lexicon data/callhome/train_4lang/lexicon.txt --beam 100
