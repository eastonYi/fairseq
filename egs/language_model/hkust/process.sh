TEXT=/data3/easton/data/HKUST/HKUST_120
TOKENIZERS_PARALLELISM=false fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/train/text.char \
    --validpref $TEXT/dev/dev.char \
    --destdir data-bin/hkust_char \
    --workers 10
