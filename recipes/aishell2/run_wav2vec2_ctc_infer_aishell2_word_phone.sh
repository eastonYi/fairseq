lm_weight=3.0
word_score=2.26
RESULT_DIR=exp/wav2vec2_base_finetune_aishell2_word_phone/decoder_result_aishell2_selflm${lm_weight}_score${word_score}
CUDA_VISIBLE_DEVICES=0
nohup python examples/speech_recognition/infer_ma.py data/aishell2/test_wav2vec2/word_phone --task audio_pretraining \
--nbest 1 --path exp/wav2vec2_base_finetune_aishell2_word_phone/checkpoint_best.pt --gen-subset "android_test" --results-path $RESULT_DIR --w2l-decoder kenlm \
--lm-model data/aishell2/train_wav2vec2_finetune/word_phone/text.aishell2.4gram.arpa.bin --lm-weight ${lm_weight} --word-score ${word_score} --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
--post-process letter  --lexicon data/aishell2/train_wav2vec2_finetune/word_phone/lexicon.txt --beam 100 > nohup.ou0 &
