#!/usr/bin/env bash

echo "Prepare train and test jsons"
# for part in train dev; do
#     python asr_prep_json.py \
#     --audio-dirs ${download_dir}/LibriSpeech/${part} \
#     --labels ${download_dir}/LibriSpeech/${part}/text \
#     --audio-format wav --dictionary ${fairseq_dict} --output ${part}.json
# done

AUDIO_DIR=/data3/easton/data/HKUST/audio/dev_wav_segs
LABELS=/data3/easton/data/HKUST/HKUST_120/test/trans.char
Dictionary=/home/easton/projects/wav2vec/egs/hkust/data/dict.char.txt
part=../data2/test

python asr_prep_json.py \
--audio-dirs $AUDIO_DIR \
--labels $LABELS \
--audio-format wav --dictionary $Dictionary --output ${part}.json

AUDIO_DIR=/data3/easton/data/HKUST/audio/train_wav_segs
LABELS=/data3/easton/data/HKUST/HKUST_120/dev/trans.char
part=../data2/dev

python asr_prep_json.py \
--audio-dirs $AUDIO_DIR \
--labels $LABELS \
--audio-format wav --dictionary $Dictionary --output ${part}.json

AUDIO_DIR=/data3/easton/data/HKUST/audio/train_wav_segs
LABELS=/data3/easton/data/HKUST/HKUST_120/train/trans.char
part=../data2/train

python asr_prep_json.py \
--audio-dirs $AUDIO_DIR \
--labels $LABELS \
--audio-format wav --dictionary $Dictionary --output ${part}.json
