lm_weight=1.46
word_score=-0.59
RESULT_DIR=exp/wav2vec2_base_finetune_hkust_word_phone_v1/decoder_result_hkust_selflm${lm_weight}_score${word_score}
CUDA_VISIBLE_DEVICES=0
python examples/speech_recognition/infer_ma.py data/hkust/test/word_phone_v1 --task audio_pretraining \
--nbest 1 --path exp/wav2vec2_base_finetune_hkust_word_phone_v1/checkpoint_best.pt --gen-subset "test" --results-path $RESULT_DIR --w2l-decoder kenlm \
--lm-model data/hkust/test/word_phone_v1/5gram.hkust.arpa.bin --lm-weight ${lm_weight} --word-score ${word_score} --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
--post-process letter  --lexicon data/hkust/test/word_phone_v1/lexicon.txt --beam 500
