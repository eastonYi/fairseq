# TEXT=/data3/easton/data/HKUST/HKUST_120
DATA_DIR=/home/easton/projects/fairseq/egs/language_model/hkust/data/hkust_char_add_callhome_ma
TOKENIZERS_PARALLELISM=false fairseq-preprocess \
    --only-source \
    --trainpref $DATA_DIR/train.char.text \
    --validpref $DATA_DIR/callhome_dev.char.text \
    --destdir $DATA_DIR \
    --workers 10
