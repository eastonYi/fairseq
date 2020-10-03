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
Dictionary=dict.char.txt
part=test

python asr_prep_json.py \
--audio-dirs $AUDIO_DIR \
--labels $LABELS \
--audio-format wav --dictionary $Dictionary --output ${part}.json

AUDIO_DIR=/data3/easton/data/HKUST/audio/train_wav_segs
LABELS=/data3/easton/data/HKUST/HKUST_120/dev/trans.char
part=dev

python asr_prep_json.py \
--audio-dirs $AUDIO_DIR \
--labels $LABELS \
--audio-format wav --dictionary $Dictionary --output ${part}.json

AUDIO_DIR=/data3/easton/data/HKUST/audio/train_wav_segs
LABELS=/data3/easton/data/HKUST/HKUST_120/train/trans.char
part=train

python asr_prep_json.py \
--audio-dirs $AUDIO_DIR \
--labels $LABELS \
--audio-format wav --dictionary $Dictionary --output ${part}.json
