DATA_DIR=/home/easton/projects/fairseq/egs/language_model/big_cn_200M/data
TOKENIZERS_PARALLELISM=false fairseq-preprocess \
    --only-source --srcdict $DATA_DIR/dict.char.txt \
    --trainpref $DATA_DIR/train_200M.char.text \
    --validpref $DATA_DIR/callhome_dev.char.text \
    --destdir $DATA_DIR \
    --workers 10
