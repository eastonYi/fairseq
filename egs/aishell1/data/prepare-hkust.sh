#!/usr/bin/env bash
AUDIO_DIR=/mnt/lustre/xushuang2/easton/data/aishell/wav
LABELS=/mnt/lustre/xushuang2/easton/projects/fairseq/egs/aishell/data
Dictionary=/home/easton/projects/wav2vec/egs/hkust/data/dict.char.txt
part=test


echo "Prepare train and test jsons"
for part in train dev; do
    python asr_prep_json.py \
    --audio-dirs ${AUDIO_DIR}/${part} \
    --labels ${LABELS}/${part}/${part}.trans \
    --audio-format wav --dictionary ${Dictionary} --output ${part}.json
done
