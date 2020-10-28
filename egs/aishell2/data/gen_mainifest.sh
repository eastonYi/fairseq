. ../path.sh
# DATA_DIR=~/data/LIBRISPEECH/train_960_flac
# DATA_DIR=~/data/LIBRISPEECH/dev-clean
DATA_DIR=$1
NAME=$2
DEST_DIR=data

python $SRC_ROOT/wav2vec/wav2vec_manifest.py \
$DATA_DIR --dest $DEST_DIR --ext wav --valid-percent 0 \
--train $NAME --valid tmp
