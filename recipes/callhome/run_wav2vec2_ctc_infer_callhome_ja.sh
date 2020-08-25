lm_weight=1.46
word_score=0.52
RESULT_DIR=exp/wav2vec2_base_finetune_callhome_phone/decoder_result_callhome_ja_selflm${lm_weight}_score${word_score}
CUDA_VISIBLE_DEVICES=5 python examples/speech_recognition/infer_ma.py data/callhome/test_400 --task audio_pretraining \
--nbest 1 --path exp/wav2vec2_base_finetune_callhome_phone/checkpoint_best.pt --gen-subset "test_ja" --results-path $RESULT_DIR --w2l-decoder kenlm \
--lm-model data/callhome/train/ja1000.arpa.bin --lm-weight ${lm_weight} --word-score ${word_score} --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
--post-process letter  --lexicon data/callhome/train/lexicon.txt --beam 100
