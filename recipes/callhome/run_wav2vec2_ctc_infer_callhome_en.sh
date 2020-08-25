lm_weight=2.46
word_score=-0.59
RESULT_DIR=exp/wav2vec2_base_finetune_callhome/decoder_result_callhome_en_selflm${lm_weight}_score${word_score}
CUDA_VISIBLE_DEVICES=0 python examples/speech_recognition/infer_en.py data/callhome/test --task audio_pretraining \
--nbest 1 --path exp/wav2vec2_base_finetune_callhome/checkpoint_best.pt --gen-subset "test_en" --results-path $RESULT_DIR --w2l-decoder kenlm \
--lm-model data/callhome/train/callhome.arpa.bin --lm-weight ${lm_weight} --word-score ${word_score} --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
--post-process letter  --lexicon data/callhome/train/lexicon.txt --beam 100
