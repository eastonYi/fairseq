. ../path.sh
# DATA_DIR=~/data/LIBRISPEECH/train_960_flac
# DATA_DIR=~/data/LIBRISPEECH/dev-clean
DATA_DIR=~/data/LIBRISPEECH/dev-other
DEST_DIR=/home/easton/projects/fairseq/egs/libri/data

python $SRC_ROOT/wav2vec/wav2vec_manifest.py \
$DATA_DIR --dest $DEST_DIR --ext flac --valid-percent 0 \
--train dev_other.tsv --valid tmp
