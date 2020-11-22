TEXT=data
TOKENIZERS_PARALLELISM=false fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/text.char\
    --validpref $TEXT/dev.text.char \
    --srcdict $TEXT/dict.char.txt \
    --destdir data-bin \
    --workers 10
